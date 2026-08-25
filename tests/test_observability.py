from maintenance_intelligence.observability import ApiMetrics, new_request_id


def test_request_id_preserves_valid_caller_id_and_generates_fallback():
    assert new_request_id("trace-123") == "trace-123"
    generated = new_request_id(None)
    assert len(generated) == 36
    assert generated.count("-") == 4


def test_prometheus_metrics_include_requests_and_answer_outcomes():
    metrics = ApiMetrics()
    metrics.requests.labels("POST", "/ask", "200").inc()
    metrics.duration.labels("POST", "/ask").observe(0.25)
    metrics.answers.labels("false").inc()
    rendered = metrics.render().decode("utf-8")
    assert 'maintenance_api_requests_total{method="POST",route="/ask",status="200"} 1.0' in rendered
    assert 'maintenance_answers_total{abstained="false"} 1.0' in rendered
