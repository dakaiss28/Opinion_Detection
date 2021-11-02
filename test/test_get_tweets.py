""" tests for the module get_tweets"""
from src.get_tweets import set_up


def test_set_up():
    """test method set_up"""
    (api, connexion) = set_up("..\\twitter_token.json.txt")
    user = api.get_user(screen_name="twitter")
    user_name = user.screen_name()
    assert user_name == "Twitter"
    cursor = connexion.cursor()
    cursor.execute(
        "INSERT INTO dbo.tweets values(?,?,?,?,?,?,?)",
        1,
        None,
        "hello",
        0,
        0,
        None,
        0,
    )
    connexion.commit()
    result = cursor.execute("Select content from dbo.tweets WHERE id = 1")
    assert result == "hello"
