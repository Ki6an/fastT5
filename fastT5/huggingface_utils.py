_auth_token = None

def set_auth_token(token):
    """Set the token which allows the user to authenticate to hugginface.co for downloading private models

    Args:
        token (Union[str, bool]): The token value to store. One of:
            - an API key (from https://huggingface.co/organizations/ORGNAME/settings/token),
            - a login token obtained by running `$ transformers-cli login`
            - `True`, which tells transformers to use the login token stored in ~/.huggingface/token

    Returns:
        None
    """
    global _auth_token
    _auth_token = token

def get_auth_token():
    """Get the user-configurable auth token, which defaults to None

    Returns:
        auth_token (Optional[Union[str, bool]]) for authenticating with huggingface.co
    """
    global _auth_token
    return _auth_token