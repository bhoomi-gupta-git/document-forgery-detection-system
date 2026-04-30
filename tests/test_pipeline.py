"""
tests/test_pipeline.py — Tests for CV/ML pipeline modules.
Run with: pytest tests/test_pipeline.py -v
"""

import numpy as np
import pytest


# ── Aggregator tests ──────────────────────────────────────────────────────────

class TestAggregator:

    def _detector_out(self, verdict='authentic', confidence=0.2, ela=0.1, detections=None):
        return {
            'verdict':      verdict,
            'confidence':   confidence,
            'heatmap_path': None,
            'detections':   detections or [],
            'ela_score':    ela,
        }

    def _ocr_out(self, text='', blocks=None):
        return {'text': text, 'blocks': blocks or []}

    def test_authentic_verdict_low_confidence(self):
        from pipeline.aggregator import aggregate
        result = aggregate(
            self._detector_out('authentic', 0.1, 0.05),
            self._ocr_out(),
            'test-uuid-001',
            processing_ms=500
        )
        assert result['verdict'] == 'authentic'
        assert result['confidence'] < 0.5

    def test_forged_verdict_high_confidence(self):
        from pipeline.aggregator import aggregate
        result = aggregate(
            self._detector_out('forged', 0.95, 0.80),
            self._ocr_out(),
            'test-uuid-002',
            processing_ms=1000
        )
        assert result['verdict'] == 'forged'
        assert result['confidence'] >= 0.5

    def test_unknown_verdict_mock_mode(self):
        from pipeline.aggregator import aggregate
        result = aggregate(
            self._detector_out('unknown', 0.0, 0.0),
            self._ocr_out(),
            'test-uuid-003',
            processing_ms=100
        )
        assert result['verdict'] == 'unknown'

    def test_result_has_required_keys(self):
        from pipeline.aggregator import aggregate
        result = aggregate(
            self._detector_out(),
            self._ocr_out(),
            'test-uuid-004',
            processing_ms=300
        )
        required = [
            'id', 'document_id', 'verdict', 'confidence',
            'detections', 'annotated_image', 'ocr_text',
            'processing_ms', 'model_version', 'analysed_at', 'error_message'
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_confidence_clamped_0_to_1(self):
        from pipeline.aggregator import aggregate
        result = aggregate(
            self._detector_out('forged', 1.0, 1.0),
            self._ocr_out(),
            'test-uuid-005',
            processing_ms=200
        )
        assert 0.0 <= result['confidence'] <= 1.0

    def test_detections_list_returned(self):
        from pipeline.aggregator import aggregate
        detections = [
            {'type': 'ela_artifact', 'confidence': 0.85,
             'region': None, 'description': 'ELA residual'}
        ]
        result = aggregate(
            self._detector_out('forged', 0.9, 0.7, detections),
            self._ocr_out(),
            'test-uuid-006',
            processing_ms=1500
        )
        assert isinstance(result['detections'], list)
        assert len(result['detections']) >= 1

    def test_annotated_image_relative_path(self):
        from pipeline.aggregator import aggregate
        det = self._detector_out()
        det['heatmap_path'] = '/absolute/path/to/results/abc_annotated.jpg'
        result = aggregate(det, self._ocr_out(), 'test-uuid-007', 200)
        # Should store only the basename
        assert result['annotated_image'] == 'abc_annotated.jpg'

    def test_ocr_text_stored(self):
        from pipeline.aggregator import aggregate
        result = aggregate(
            self._detector_out(),
            self._ocr_out(text='Invoice No: 1042'),
            'test-uuid-008',
            processing_ms=400
        )
        assert result['ocr_text'] == 'Invoice No: 1042'


# ── OCR consistency heuristic ─────────────────────────────────────────────────

class TestOcrConsistency:

    def test_empty_blocks_returns_zero(self):
        from pipeline.aggregator import _compute_ocr_consistency
        assert _compute_ocr_consistency({'text': '', 'blocks': []}) == 0.0

    def test_few_blocks_returns_zero(self):
        from pipeline.aggregator import _compute_ocr_consistency
        # Fewer than 3 blocks → 0.0
        blocks = [{'text': 'a', 'confidence': 90}]
        assert _compute_ocr_consistency({'text': 'a', 'blocks': blocks}) == 0.0

    def test_uniform_confidence_low_score(self):
        from pipeline.aggregator import _compute_ocr_consistency
        # All same confidence → stdev=0 → score=0
        blocks = [{'text': 'word', 'confidence': 90}] * 10
        score = _compute_ocr_consistency({'text': 'word ' * 10, 'blocks': blocks})
        assert score == 0.0

    def test_high_variance_gives_nonzero_score(self):
        from pipeline.aggregator import _compute_ocr_consistency
        blocks = (
            [{'text': 'good', 'confidence': 95}] * 5 +
            [{'text': '???',  'confidence': 10}] * 5
        )
        score = _compute_ocr_consistency({'text': '', 'blocks': blocks})
        assert score > 0.0


# ── Detector mock fallback ────────────────────────────────────────────────────

class TestDetectorMock:

    def test_mock_result_structure(self):
        from pipeline.detector import _mock_result
        result = _mock_result(ela_score=0.2)
        assert result['verdict']      == 'unknown'
        assert result['confidence']   == 0.0
        assert result['heatmap_path'] is None
        assert isinstance(result['detections'], list)

    def test_ela_score_blank_image(self):
        from pipeline.detector import _compute_ela_score
        blank = np.zeros((224, 224, 3), dtype=np.uint8)
        score = _compute_ela_score(blank)
        assert 0.0 <= score <= 0.1

    def test_ela_score_range(self):
        from pipeline.detector import _compute_ela_score
        # Any image should give score in [0, 1]
        random_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        score = _compute_ela_score(random_img)
        assert 0.0 <= score <= 1.0

    def test_build_detections_empty_when_low_scores(self):
        from pipeline.detector import _build_detections
        detections = _build_detections(cnn_confidence=0.1, ela_score=0.05)
        assert detections == []

    def test_build_detections_ela_artifact_detected(self):
        from pipeline.detector import _build_detections
        detections = _build_detections(cnn_confidence=0.1, ela_score=0.8)
        types = [d['type'] for d in detections]
        assert 'ela_artifact' in types

    def test_build_detections_pattern_anomaly_when_forged(self):
        from pipeline.detector import _build_detections
        detections = _build_detections(cnn_confidence=0.9, ela_score=0.1)
        types = [d['type'] for d in detections]
        assert 'pattern_anomaly' in types


# ── DB adapter tests ──────────────────────────────────────────────────────────

class TestDbAdapter:

    def _make_doc(self, doc_id='test-doc-1'):
        from datetime import datetime, timezone
        return {
            'id':              doc_id,
            'filename_orig':   'invoice.jpg',
            'filename_stored': f'{doc_id}.jpg',
            'file_ext':        'jpg',
            'file_size_bytes': 204800,
            'mime_type':       'image/jpeg',
            'uploaded_at':     datetime.now(timezone.utc).isoformat(),
            'status':          'pending',
        }

    def test_insert_and_get_document(self):
        from db import adapter as db
        doc = self._make_doc('db-test-001')
        db.insert_document(doc)
        row = db.get_document('db-test-001')
        assert row is not None
        assert row['filename_orig'] == 'invoice.jpg'

    def test_update_document_status(self):
        from db import adapter as db
        doc = self._make_doc('db-test-002')
        db.insert_document(doc)
        db.update_document_status('db-test-002', 'complete')
        row = db.get_document('db-test-002')
        assert row['status'] == 'complete'

    def test_get_nonexistent_document_returns_none(self):
        from db import adapter as db
        row = db.get_document('does-not-exist')
        assert row is None

    def test_get_history_empty(self):
        from db import adapter as db
        result = db.get_history()
        assert result['total'] == 0
        assert result['items'] == []

    def test_get_history_pagination_keys(self):
        from db import adapter as db
        result = db.get_history(page=1, limit=10)
        assert 'page'  in result
        assert 'limit' in result
        assert 'total' in result
        assert 'items' in result

    def test_insert_result_and_retrieve(self):
        from db import adapter as db
        from datetime import datetime, timezone
        import uuid

        doc = self._make_doc('db-test-003')
        db.insert_document(doc)
        db.update_document_status('db-test-003', 'complete')

        result = {
            'id':              str(uuid.uuid4()),
            'document_id':     'db-test-003',
            'verdict':         'authentic',
            'confidence':      0.92,
            'detections':      [],
            'annotated_image': None,
            'ocr_text':        'Sample text',
            'processing_ms':   3200,
            'model_version':   '1.0.0',
            'analysed_at':     datetime.now(timezone.utc).isoformat(),
            'error_message':   None,
        }
        db.insert_result(result)

        fetched = db.get_result('db-test-003')
        assert fetched is not None
        assert fetched['verdict']    == 'authentic'
        assert fetched['confidence'] == 0.92
        assert isinstance(fetched['detections'], list)