from io import StringIO


def capture_schema(df, name="DataFrame"):
    buf = StringIO()
    df.printSchema(printTreeString=lambda x: buf.write(x + "\n"))
    logger.info(f"{name} schema:\n{buf.getvalue()}")


def capture_show(df, name="DataFrame", n=5, truncate=False):
    buf = StringIO()
    print(df.select("*").limit(n)._jdf.showString(n, int(truncate)), file=buf)
    logger.info(f"{name} preview:\n{buf.getvalue()}")


def inference_wrapper(base64_str, image_id):
    score = inference(base64_str, image_id)
    if score == -1.0:
        logger.warning(f"Inference failed for ID {image_id}")
    return score
