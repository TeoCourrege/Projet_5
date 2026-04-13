import pytest
import pandas as pd
import os
import tempfile
from unittest.mock import patch, MagicMock


def test_gradio_app_loads():
    """Verify the Gradio Blocks app object builds without errors."""
    with patch("src.db.database.init_db"), \
         patch("src.db.database.seed_admin"), \
         patch("src.db.database.joblib") as mock_jl:
        mock_jl.load.return_value = MagicMock()
        import importlib
        import app as app_module
        importlib.reload(app_module)
        assert app_module.demo is not None


def test_authenticate_valid_user():
    """Mocked authentication returns True for valid credentials."""
    from src.db.database import hash_password, verify_password

    hashed = hash_password("test123")
    assert verify_password("test123", hashed) is True


def test_authenticate_invalid_password():
    """verify_password rejects wrong password."""
    from src.db.database import hash_password, verify_password

    hashed = hash_password("correct")
    assert verify_password("wrong", hashed) is False


def test_batch_predict_rejects_bad_format():
    """batch_predict returns error message for unsupported file format."""
    from unittest.mock import MagicMock

    fake_file = MagicMock()
    fake_file.name = "data.xml"

    with patch("src.db.database._get_pipeline") as mock_pipe, \
         patch("src.db.database.SessionLocal") as mock_sess:
        mock_sess.return_value = MagicMock()
        from src.db.database import batch_predict
        result = batch_predict(fake_file)
        assert "non supporté" in result


def test_batch_predict_requires_id_column():
    """batch_predict returns error when 'id' column is missing."""
    from unittest.mock import MagicMock

    df = pd.DataFrame({"age": [30], "genre": ["M"]})

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f, index=False)
        tmp_path = f.name

    try:
        fake_file = MagicMock()
        fake_file.name = tmp_path

        with patch("src.db.database._get_pipeline"), \
             patch("src.db.database.SessionLocal") as mock_sess:
            mock_sess.return_value = MagicMock()
            from src.db.database import batch_predict
            result = batch_predict(fake_file)
            assert "id" in result
    finally:
        os.unlink(tmp_path)
