def success(data=None, message="Success"):
    return {
        "status": True,
        "message": message,
        "data": data
    }

def error(message="Error"):
    return {
        "status": False,
        "message": message
    }