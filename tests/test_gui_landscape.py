"""The GUI keeps separate full-resolution outputs for selected methods."""

import os
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets

import src.settings as settings
from src.MainWindow import Window


def test_gui_multiselect_results_and_selection(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    settings.init()
    settings.globalVars["QSettings"] = QtCore.QSettings(
        str(tmp_path / "settings.ini"), QtCore.QSettings.IniFormat
    )
    settings.globalVars["MainApplication"] = app
    with tempfile.TemporaryDirectory() as preview_dir:
        settings.globalVars["RootTempDir"] = type("Temp", (), {"name": preview_dir})()
        window = Window()
        paths = ["tests/low_res_images/DSC_0356.jpg", "tests/low_res_images/DSC_0358.jpg"]
        window.set_new_loaded_image_files(paths)
        buttons = window.SettingsWidget._method_buttons
        assert "near_far_cut" in buttons
        assert "landscape_regions" in buttons
        buttons["landscape"].click()
        buttons["laplacian"].click()
        buttons["landscape_blend"].click()
        assert window.SettingsWidget.get_selected_stacking_methods() == [
            "landscape", "landscape_blend"
        ]
        assert window.SettingsWidget.get_algorithm_config().alignment_mode == "auto"
        assert window.stack_loaded_images()
        deadline = time.monotonic() + 40
        while window.is_stacking and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()
        assert not window.is_stacking
        assert window._stack_error is None
        assert len(window._results) == 2
        paths = list(window._results)
        result_list = window._main_content.ImageWidgets.processed_images_widget.list
        result_list.setCurrentRow(0)
        app.processEvents()
        assert window._selected_result == paths[0]
        first = window.LaplacianAlgorithm.output_image
        result_list.setCurrentRow(1)
        app.processEvents()
        assert window._selected_result == paths[1]
        second = window.LaplacianAlgorithm.output_image
        assert first is not second
        assert first.dtype == second.dtype
        assert window.has_unsaved_work
        window._exported_results.update(paths)
        window._main_content._preview_pool.waitForDone()
        app.processEvents()
        window.close()
