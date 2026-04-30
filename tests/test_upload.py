"""
tests/test_upload.py — Tests for file validation and upload route.
Run with: pytest tests/test_upload.py -v
"""

import io
import pytest


# ── Validator unit tests ──────────────────────────────────────────────────────

class TestValidatorExtension:
    def _fake_file(self, filename, content=b'\xff\xd8\xff\x00' * 10, size=None):
        """Helper: create a minimal fake FileStorage-like object."""
        class FakeFile:
            def __init__(self, name, data):
                self.filename = name
                self._data = data
                self._pos  = 0
            def read(self, n=-1):
                chunk = self._data[self._pos:self._pos + (n if n > 0 else len(self._data))]
                self._pos += len(chunk)
                return chunk
            def seek(self, pos, whence=0):
                if whence == 0: self._pos = pos
                elif whence == 2: self._pos = len(self._data) + pos
            def tell(self):
                return self._pos if hasattr(self, '_pos') else 0
        return FakeFile(filename, content)

    def test_valid_jpg_passes(self):
        from utils.validator import validate_upload
        f = self._fake_file('scan.jpg')
        # Extension passes; MIME check may vary without python-magic — just test extension gate
        result = validate_upload(f)
        # Should not fail on extension
        assert 'extension' not in (result.error or '').lower()

    def test_exe_extension_rejected(self):
        from utils.validator import validate_upload
        f = self._fake_file('virus.exe', b'MZ\x90\x00' * 10)
        result = validate_upload(f)
        assert result.ok is False
        assert 'not supported' in result.error.lower()

    def test_no_file_rejected(self):
        from utils.validator import validate_upload
        result = validate_upload(None)
        assert result.ok is False

    def test_empty_filename_rejected(self):
        from utils.validator import validate_upload
        class NoName:
            filename = ''
            def read(self, n): return b''
            def seek(self, *a): pass
            def tell(self): return 0
        result = validate_upload(NoName())
        assert result.ok is False

    def test_oversized_file_rejected(self):
        from utils.validator import validate_upload
        import config

        class BigFile:
            filename = 'big.jpg'
            def read(self, n): return b'\xff\xd8\xff' + b'\x00' * min(n, 2048)
            def seek(self, pos, whence=0): pass
            def tell(self): return config.MAX_CONTENT_LENGTH + 1024

        result = validate_upload(BigFile())
        assert result.ok is False
        assert 'exceeds' in result.error.lower()

    def test_empty_file_rejected(self):
        from utils.validator import validate_upload

        class Empty:
            filename = 'empty.png'
            def read(self, n): return b''
            def seek(self, pos, whence=0): pass
            def tell(self): return 0

        result = validate_upload(Empty())
        assert result.ok is False
        assert 'empty' in result.error.lower()


# ── File extension helper ─────────────────────────────────────────────────────

class TestGetExtension:
    def test_jpg(self):
        from utils.validator import _get_extension
        assert _get_extension('photo.jpg') == 'jpg'

    def test_jpeg_normalised(self):
        from utils.validator import _get_extension
        assert _get_extension('scan.JPEG') == 'jpeg'

    def test_pdf(self):
        from utils.validator import _get_extension
        assert _get_extension('document.pdf') == 'pdf'

    def test_no_extension(self):
        from utils.validator import _get_extension
        assert _get_extension('noext') == ''


# ── File size formatting ──────────────────────────────────────────────────────

class TestGetMetadata:
    def test_metadata_keys_present(self):
        from utils.validator import get_file_metadata

        class FakeFile:
            filename = 'test.png'
            def read(self, n): return b'\x89PNG' + b'\x00' * max(0, n - 4)
            def seek(self, pos, whence=0): pass
            def tell(self): return 1024

        meta = get_file_metadata(FakeFile())
        assert 'filename_orig'   in meta
        assert 'file_ext'        in meta
        assert 'file_size_bytes' in meta
        assert 'mime_type'       in meta

    def test_jpeg_extension_normalised_to_jpg(self):
        from utils.validator import get_file_metadata

        class FakeFile:
            filename = 'scan.jpeg'
            def read(self, n): return b'\xff\xd8\xff' + b'\x00' * max(0, n - 3)
            def seek(self, pos, whence=0): pass
            def tell(self): return 500

        meta = get_file_metadata(FakeFile())
        assert meta['file_ext'] == 'jpg'


# ── Upload route integration tests ────────────────────────────────────────────

class TestUploadRoute:
    def test_upload_no_file_returns_400(self, client):
        resp = client.post('/api/upload', data={})
        assert resp.status_code == 400
        data = resp.get_json()
        assert 'error' in data

    def test_upload_wrong_type_returns_400(self, client):
        data = {
            'file': (io.BytesIO(b'MZ\x90\x00' * 100), 'malware.exe')
        }
        resp = client.post(
            '/api/upload',
            data=data,
            content_type='multipart/form-data'
        )
        assert resp.status_code == 400
        body = resp.get_json()
        assert 'error' in body

    def test_history_route_returns_200(self, client):
        resp = client.get('/api/history')
        assert resp.status_code == 200
        body = resp.get_json()
        assert 'items' in body
        assert 'total' in body

    def test_result_not_found_returns_404(self, client):
        resp = client.get('/api/result/nonexistent-uuid')
        assert resp.status_code == 404

    def test_upload_screen_loads(self, client):
        resp = client.get('/')
        assert resp.status_code == 200
        assert b'DocForge' in resp.data

    def test_history_screen_loads(self, client):
        resp = client.get('/history')
        assert resp.status_code == 200

    def test_404_screen(self, client):
        resp = client.get('/this-page-does-not-exist')
        assert resp.status_code == 404