"""cp_ prefix: COMPACK worker. Calls mxtaltools' own single_compack_run (PackingSimilarity, packing_shell_size 20,
engine defaults otherwise), so the settings are the pipeline's, not a copy. A (0, 0) return is that function's
AttributeError sentinel (engine failure), recorded as status 'failed'; any other exception is recorded, not raised."""
import contextlib
import io
import time


def work(args):
    key, ref_path, test_path = args
    t = time.time()
    from mxtaltools.dataset_utils.data_class_methods.crystal_analysis import single_compack_run
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rmsd, nm = single_compack_run(key, test_path, ref_path)
        status = 'failed' if (nm == 0 and rmsd == 0) else 'ok'
        return key, float(rmsd), int(nm), status, time.time() - t, buf.getvalue()[-300:]
    except Exception as e:  # noqa: BLE001 - recorded, never silently read as a result
        return key, float('nan'), -1, f"error {type(e).__name__}: {str(e)[:200]}", time.time() - t, buf.getvalue()[-300:]
