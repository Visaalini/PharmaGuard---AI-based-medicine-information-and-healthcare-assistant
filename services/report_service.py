def generate_report(result):

    report = {
        "medicine_status": result["prediction"],
        "details": result["text"]
    }

    return report