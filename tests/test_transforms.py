from esp_data.dataset.transforms import SimpleTransformStep, TransformPipeline, TransformStep


def add_int(a: int, b: int) -> int:
    return a + b


def test_simple_transform():
    step = SimpleTransformStep(
        function=add_int,
        function_name="add_int",
        parameters={"b": 1},
    )
    assert step.version == "unknown"

    # test call
    data = 0
    data = step(data)
    assert data == 1

    step_dict = step.to_dict()

    new_step = SimpleTransformStep.from_dict(step_dict)

    # apply again
    data = new_step(data)
    assert data == 2


def test_transform_step():
    step = TransformStep(
        module_path="esp_data.dataset.transforms",
        function_name="test_method_add_int",
        parameters={"b": 1},
    )
    assert step.version != "unknown"

    # test call
    data = 0
    data = step(data)
    assert data == 1

    step_dict = step.to_dict()

    new_step = TransformStep.from_dict(step_dict)

    # apply again
    data = new_step(data)
    assert data == 2


def test_transform_pipeline(tmp_path):
    pipeline = TransformPipeline(
        name="test_pipeline",
        steps=[
            TransformStep(
                module_path="esp_data.dataset.transforms",
                function_name="test_method_add_int",
                parameters={"b": 1},
            ),
            TransformStep(
                module_path="esp_data.dataset.transforms",
                function_name="test_method_multiply",
                parameters={"b": 2},
            ),
            TransformStep(
                module_path="esp_data.dataset.transforms",
                function_name="multiply",
                parameters={},
                is_class_method=True,
                class_name="TestClass",
                init_parameters={"factor": 2},
            ),
        ],
    )

    # test call
    data = 0
    data = pipeline(data)
    assert data == 4

    pipeline_dict = pipeline.to_dict()

    new_pipeline = TransformPipeline.from_dict(pipeline_dict)

    # apply again
    data = new_pipeline(data)
    assert data == 20

    new_pipeline.save(tmp_path / "pipeline.json")

    another_pipeline = TransformPipeline.load(tmp_path / "pipeline.json")

    # apply again
    data = another_pipeline(data)
    assert data == 84
