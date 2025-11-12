```console
aide data_dir="data/micromodels" goal="Your task is to predict the emotional reaction level of a given response. You can only use the corresponding rationales for the train data, but not for the test data. You are not allowed to choose the final model yourself, instead you must use an ExplainableBoostingClassifier from the interpret.glassbox package." eval="micro F1 score" agent.code.model="gpt-4.1"
```
```console
Traceback (most recent call last):
  File "/Users/XXX/projects/aide/venv/bin/aide", line 8, in <module>
    sys.exit(run())
             ^^^^^
  File "/Users/XXX/projects/aide/venv/lib/python3.12/site-packages/aide/run.py", line 131, in run
    agent.step(exec_callback=exec_callback)
  File "/Users/XXX/projects/aide/venv/lib/python3.12/site-packages/aide/agent.py", line 284, in step
    result_node = self._draft()
                  ^^^^^^^^^^^^^
  File "/Users/XXX/projects/aide/venv/lib/python3.12/site-packages/aide/agent.py", line 204, in _draft
    plan, code = self.plan_and_code_query(prompt)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/XXX/projects/aide/venv/lib/python3.12/site-packages/aide/agent.py", line 157, in plan_and_code_query
    completion_text = query(
                      ^^^^^^
  File "/Users/XXX/projects/aide/venv/lib/python3.12/site-packages/aide/backend/__init__.py", line 45, in query
    output, req_time, in_tok_count, out_tok_count, info = query_func(
                                                          ^^^^^^^^^^^
  File "/Users/XXX/projects/aide/venv/lib/python3.12/site-packages/aide/backend/backend_openai.py", line 83, in query
    completion = backoff_create(
                 ^^^^^^^^^^^^^^^
  File "/Users/XXX/projects/aide/venv/lib/python3.12/site-packages/backoff/_sync.py", line 48, in retry
    ret = target(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/XXX/projects/aide/venv/lib/python3.12/site-packages/aide/backend/utils.py", line 26, in backoff_create
    return create_fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/XXX/projects/aide/venv/lib/python3.12/site-packages/openai/_utils/_utils.py", line 286, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/Users/XXX/projects/aide/venv/lib/python3.12/site-packages/openai/resources/chat/completions/completions.py", line 1147, in create
    return self._post(
           ^^^^^^^^^^^
  File "/Users/XXX/projects/aide/venv/lib/python3.12/site-packages/openai/_base_client.py", line 1259, in post
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/XXX/projects/aide/venv/lib/python3.12/site-packages/openai/_base_client.py", line 1047, in request
    raise self._make_status_error_from_response(err.response) from None
openai.BadRequestError: Error code: 400 - {'error': {'message': "This model's maximum context length is 1047576 tokens. However, your messages resulted in 1408568 tokens. Please reduce the length of the messages.", 'type': 'invalid_request_error', 'param': 'messages', 'code': 'context_length_exceeded'}}
```