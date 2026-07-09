"""Headless unit tests for measurement_ocr.

No GUI / display is exercised -- these cover the pure logic that the
interactive layer depends on: geometry, the grid + field-column expansion,
the corner-grab hit-testing used while dragging, the CSV/XLSX loggers, and
the robustness guards that keep a bad corner selection or a stray drag from
crashing the app.

Run:  python -m unittest test_measurement_ocr -v
"""
import os
import tempfile
import unittest

import cv2
import numpy as np

import measurement_ocr as m


class TestGeometry(unittest.TestCase):
    def test_order_points_orders_tl_tr_br_bl(self):
        # deliberately scrambled input
        pts = np.array([[200, 150], [50, 50], [50, 150], [200, 50]], dtype=np.float32)
        tl, tr, br, bl = m.order_points(pts)
        self.assertTrue(np.allclose(tl, [50, 50]))
        self.assertTrue(np.allclose(tr, [200, 50]))
        self.assertTrue(np.allclose(br, [200, 150]))
        self.assertTrue(np.allclose(bl, [50, 150]))

    def test_canonical_size_clamped_min(self):
        # a near-zero (degenerate) quad must not yield a 0-size canvas
        q = np.array([[10, 10], [11, 10], [11, 11], [10, 11]], dtype=np.float32)
        w, h = m.canonical_size_from_quad(q)
        self.assertGreaterEqual(w, 60)
        self.assertGreaterEqual(h, 40)

    def test_canonical_size_clamped_max(self):
        # a far-dragged corner must not ask for an absurdly huge rectified image
        q = np.array([[0, 0], [999999, 0], [999999, 999999], [0, 999999]],
                     dtype=np.float32)
        w, h = m.canonical_size_from_quad(q)
        self.assertLessEqual(w, m.CANONICAL_MAX)
        self.assertLessEqual(h, m.CANONICAL_MAX)

    def test_degenerate_quad_does_not_crash_pipeline(self):
        # all-identical corners: canonical size stays bounded and no throw
        q = m.order_points(np.array([[80, 80]] * 4, dtype=np.float32))
        w, h = m.canonical_size_from_quad(q)
        self.assertLessEqual(max(w, h), m.CANONICAL_MAX)


class TestGrid(unittest.TestCase):
    def test_parse_valid(self):
        self.assertEqual(m.parse_grid_spec("3x4"), (3, 4))
        self.assertEqual(m.parse_grid_spec("2X2"), (2, 2))
        self.assertEqual(m.parse_grid_spec("3*3"), (3, 3))
        self.assertEqual(m.parse_grid_spec("5,1"), (5, 1))

    def test_parse_invalid(self):
        for bad in ("3", "axb", "0x4", "2x0", ""):
            with self.assertRaises(ValueError):
                m.parse_grid_spec(bad)

    def test_build_grid_tiles_completely(self):
        rows, cols, W, H = 3, 4, 400, 300
        fs = m.build_grid_fields(rows, cols, W, H)
        self.assertEqual(len(fs), rows * cols)
        self.assertEqual(fs[0].name, "r1c1")
        self.assertEqual(fs[-1].name, "r3c4")
        # bottom-right cell reaches the far edge (no gap / no overflow)
        x, y, w, h = fs[-1].rect
        self.assertEqual(x + w, W)
        self.assertEqual(y + h, H)
        # every cell is non-empty
        self.assertTrue(all(w > 0 and h > 0 for (_x, _y, w, h) in (f.rect for f in fs)))


class TestCornerGrab(unittest.TestCase):
    def setUp(self):
        self.f = m.Field(name="c", rect=(100, 100, 50, 40))  # TL(100,100) BR(150,140)

    def test_grab_returns_opposite_anchor(self):
        # grab near BR -> anchor is TL
        g = m._grab_field_corner([self.f], 151, 141, tol=10)
        self.assertIsNotNone(g)
        _, ax, ay = g
        self.assertEqual((ax, ay), (100, 100))
        # grab near TL -> anchor is BR
        _, ax, ay = m._grab_field_corner([self.f], 99, 99, tol=10)
        self.assertEqual((ax, ay), (150, 140))

    def test_no_grab_when_far(self):
        self.assertIsNone(m._grab_field_corner([self.f], 500, 500, tol=10))

    def test_nearest_field_wins(self):
        g = m.Field(name="g", rect=(0, 0, 10, 10))
        # cursor exactly on g's BR corner (10,10); should pick g, anchor (0,0)
        res = m._grab_field_corner([self.f, g], 10, 10, tol=6)
        self.assertIs(res[0], g)


class TestFieldColumns(unittest.TestCase):
    def test_single_value_field_one_column(self):
        f = m.Field(name="volts", rect=(0, 0, 10, 10),
                    last_tokens=["12.3"], last_boxes=[(0, 0, 5, 5)])
        names, values = m.field_columns([f])
        self.assertEqual(names, ["volts"])
        self.assertEqual(values, ["12.3"])

    def test_multi_value_field_splits_row_major(self):
        # two rows of two numbers; boxes given out of reading order on purpose
        f = m.Field(
            name="p", rect=(0, 0, 100, 100),
            last_tokens=["BL", "TR", "TL", "BR"],
            last_boxes=[(0, 50, 10, 10), (50, 0, 10, 10),
                        (0, 0, 10, 10), (50, 50, 10, 10)],
        )
        names, values = m.field_columns([f])
        self.assertEqual(names, ["p_1", "p_2", "p_3", "p_4"])
        # reading order: TL, TR, BL, BR
        self.assertEqual(values, ["TL", "TR", "BL", "BR"])

    def test_blank_field_keeps_its_column(self):
        f = m.Field(name="x", rect=(0, 0, 10, 10))  # no tokens
        names, values = m.field_columns([f])
        self.assertEqual(names, ["x"])
        self.assertEqual(values, [""])


class TestStableColumns(unittest.TestCase):
    """A number must always land in the same column, whatever order OCR
    returns the tokens in and even if some are dropped -- the regression that
    caused values to swap columns between rows in the CSV."""

    def _panel_field(self, tokens_boxes):
        f = m.Field(name="d", rect=(0, 0, 300, 300))
        f.last_tokens = [t for t, _b in tokens_boxes]
        f.last_boxes = [b for _t, b in tokens_boxes]
        return f

    def test_multirow_maps_to_row_col_cells(self):
        # 2 rows x 2 cols; boxes centered at (col*100, row*100)
        f = self._panel_field([
            ("a", (0, 0, 20, 20)), ("b", (100, 0, 20, 20)),
            ("c", (0, 100, 20, 20)), ("d", (100, 100, 20, 20)),
        ])
        names, values = m.stable_columns([f], {})
        self.assertEqual(names, ["d_r1c1", "d_r1c2", "d_r2c1", "d_r2c2"])
        self.assertEqual(values, ["a", "b", "c", "d"])

    def test_reordered_tokens_keep_their_columns(self):
        cells = [("L1", (0, 0, 30, 20)), ("L2", (100, 0, 20, 20)),
                 ("L3", (200, 0, 30, 20))]
        layout = {}
        base = dict(zip(*m.stable_columns([self._panel_field(cells)], layout)))
        # same numbers, reversed detection order + small y jitter -> same layout
        rev = [("L3", (200, 3, 30, 20)), ("L1", (0, -2, 30, 20)),
               ("L2", (100, 2, 20, 20))]
        names, values = m.stable_columns([self._panel_field(rev)], layout)
        self.assertEqual(dict(zip(names, values)), base)

    def test_dropped_token_blanks_only_its_own_cell(self):
        cells = [("L1", (0, 0, 30, 20)), ("L2", (100, 0, 20, 20)),
                 ("L3", (200, 0, 30, 20))]
        layout = {}
        m.stable_columns([self._panel_field(cells)], layout)   # lock 3 columns
        # a frame that misses the middle value must NOT shift L3 into L2's slot
        missing = [("L1", (0, 0, 30, 20)), ("L3", (200, 0, 30, 20))]
        names, values = m.stable_columns([self._panel_field(missing)], layout)
        d = dict(zip(names, values))
        self.assertEqual(d["d_1"], "L1")
        self.assertEqual(d["d_2"], "")     # middle blank
        self.assertEqual(d["d_3"], "L3")   # stayed put

    def test_single_value_field_unchanged(self):
        f = m.Field(name="volts", rect=(0, 0, 10, 10),
                    last_tokens=["12.3"], last_boxes=[(0, 0, 5, 5)])
        self.assertEqual(m.stable_columns([f], {}), (["volts"], ["12.3"]))


class TestLoggers(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def _rows_csv(self, path):
        import csv
        with open(path, newline="") as fh:
            return list(csv.reader(fh))

    def test_csv_header_then_rows(self):
        path = os.path.join(self.dir, "out.csv")
        lg = m.CsvLogger(path)
        self.assertIsNone(lg.existing_header())
        lg.append(["timestamp_s", "frame_idx", "a", "b"])
        lg.append(["1.0", 1, "10", "20"])
        lg.close()
        rows = self._rows_csv(path)
        self.assertEqual(rows[0], ["timestamp_s", "frame_idx", "a", "b"])
        self.assertEqual(rows[1], ["1.0", "1", "10", "20"])
        # reopening sees the existing header
        lg2 = m.CsvLogger(path)
        try:
            self.assertEqual(lg2.existing_header()[2:], ["a", "b"])
        finally:
            lg2.close()

    def test_xlsx_no_phantom_leading_row_and_numeric(self):
        openpyxl = __import__("openpyxl")
        path = os.path.join(self.dir, "out.xlsx")
        lg = m.XlsxLogger(path)
        self.assertIsNone(lg.existing_header())  # fresh file: no header yet
        lg.append(["timestamp_s", "frame_idx", "a", "b"])
        lg.append(["1.000", 1, "221.70", "0.00"])
        lg.close()
        ws = openpyxl.load_workbook(path).active
        rows = list(ws.iter_rows(values_only=True))
        self.assertEqual(len(rows), 2, "no phantom empty row before the header")
        self.assertEqual(rows[0], ("timestamp_s", "frame_idx", "a", "b"))
        # numeric-looking cells stored as real numbers, not text
        self.assertEqual(rows[1][2], 221.7)
        self.assertEqual(rows[1][3], 0)
        # appending again adopts the existing header
        self.assertEqual(list(m.XlsxLogger(path).existing_header())[2:], ["a", "b"])

    def test_build_row_logger_picks_by_extension(self):
        csv_lg = m.build_row_logger(os.path.join(self.dir, "a.csv"))
        xlsx_lg = m.build_row_logger(os.path.join(self.dir, "a.xlsx"))
        try:
            self.assertIsInstance(csv_lg, m.CsvLogger)
            self.assertIsInstance(xlsx_lg, m.XlsxLogger)
        finally:
            csv_lg.close()
            xlsx_lg.close()


class _FakeCap:
    def get(self, _prop):
        return 12480.0


class _RaisingLogger(m.RowLogger):
    """A logger whose writes fail, as if the output file were locked by
    another program (Excel) mid-run."""
    def existing_header(self):
        return None

    def append(self, cells):
        raise PermissionError(13, "Permission denied")


class TestOutputRobustness(unittest.TestCase):
    """Covers the failure mode the first test pass missed: the output file
    being unwritable / locked (open in Excel), which surfaces as OSError."""

    def test_unwritable_output_raises_oserror_not_generic(self):
        # Opening a *directory* as a file raises PermissionError on Windows /
        # IsADirectoryError on POSIX -- both OSError, which main now catches.
        d = tempfile.mkdtemp()
        with self.assertRaises(OSError):
            m.build_row_logger(d)   # path is a directory, cannot be opened as a file

    def test_log_row_survives_a_locked_output(self):
        f = m.Field(name="v", rect=(0, 0, 10, 10),
                    last_tokens=["12.3"], last_boxes=[(0, 0, 5, 5)])
        log_state = {}
        # Must NOT raise, even though every append() fails.
        m.log_row(_RaisingLogger(), [f], _FakeCap(), frame_idx=1, log_state=log_state)
        self.assertEqual(log_state.get("rows_logged", 0), 0)  # nothing counted as written
        self.assertTrue(log_state.get("warned_write"))        # warned once

    def test_maybe_auto_log_survives_a_locked_output(self):
        f = m.Field(name="v", rect=(0, 0, 10, 10),
                    last_tokens=["9.9"], last_boxes=[(0, 0, 5, 5)])
        ctx = {"fields": [f], "logger": _RaisingLogger(), "cap": _FakeCap(),
               "frame_idx": 3, "log_state": {}}
        m.maybe_auto_log(ctx)  # should swallow the write error, not crash
        self.assertEqual(ctx["log_state"].get("rows_logged", 0), 0)


class TestCompositeCallbackEditing(unittest.TestCase):
    """Drive the Rectified-window mouse callback with synthetic events to
    verify corner-drag editing and that gestures always end cleanly."""
    def _make(self, fields):
        state = {}
        cb = m.make_composite_mouse_callback(
            state, scale_ref=[2.0], image_h_ref=[600], buttons_ref=[[]],
            fields_ref=[fields], canon_ref=[400, 300])
        return state, cb

    def test_corner_drag_resizes_field(self):
        f = m.Field(name="cell", rect=(100, 100, 50, 40))  # BR canonical(150,140)->display(300,280)
        state, cb = self._make([f])
        cb(cv2.EVENT_LBUTTONDOWN, 300, 280, 0, None)   # grab BR
        self.assertIsNotNone(state.get("edit_field"))
        cb(cv2.EVENT_MOUSEMOVE, 360, 300, 0, None)     # -> canonical (180,150)
        cb(cv2.EVENT_LBUTTONUP, 360, 300, 0, None)
        self.assertEqual(f.rect, (100, 100, 80, 50))
        self.assertIsNone(state.get("edit_field"))

    def test_empty_space_drag_creates_new_field(self):
        state, cb = self._make([])
        cb(cv2.EVENT_LBUTTONDOWN, 10, 10, 0, None)
        self.assertTrue(state.get("dragging"))
        cb(cv2.EVENT_MOUSEMOVE, 80, 80, 0, None)
        cb(cv2.EVENT_LBUTTONUP, 80, 80, 0, None)
        self.assertIn("completed_rect", state)

    def test_release_over_toolbar_clears_edit_state(self):
        # regression: a corner drag released over the toolbar must not leave
        # the field "stuck" to the cursor.
        f = m.Field(name="cell", rect=(100, 100, 50, 40))
        state, cb = self._make([f])
        cb(cv2.EVENT_LBUTTONDOWN, 300, 280, 0, None)   # grab BR (in image area)
        self.assertIsNotNone(state.get("edit_field"))
        cb(cv2.EVENT_LBUTTONUP, 50, 620, 0, None)      # release over toolbar (y>=600)
        self.assertIsNone(state.get("edit_field"))
        self.assertFalse(state.get("dragging"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
