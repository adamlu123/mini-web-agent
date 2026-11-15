BENCHCAD_COMMIT_LABEL = "org.benchcad.commit"


def validate_image(
    client,
    image: str,
    benchmark_commit: str,
) -> str:
    image_object = client.images.get(image)
    labels = image_object.attrs.get("Config", {}).get("Labels") or {}
    actual = labels.get(BENCHCAD_COMMIT_LABEL)
    if actual != benchmark_commit:
        raise RuntimeError(
            f"Container image {image!r} has BenchCAD commit label "
            f"{actual!r}; expected {benchmark_commit!r}"
        )
    return str(image_object.id)
