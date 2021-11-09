""" tests for the module tweets_classification"""
from tweets_classification import set_up


def test_set_up():
    """test method set_up"""
    (api, connexion) = set_up("../twitter_token.json.txt")
    user = api.get_user(screen_name="twitter")
    user_name = user.screen_name()
    assert user_name == "Twitter"
    cursor = connexion.cursor()
    cursor.execute(
        "INSERT INTO dbo.tweets values(?,?,?,?,?,?)",
        1,
        None,
        "hello",
        0,
        0,
        None,
    )
    connexion.commit()
    result = cursor.execute("Select content from dbo.tweets WHERE id = 1")
    print(result)
    assert result == "hello"
    cursor.execute("delete * from dbo.tweets WHERE id = 1")
    connexion.commit


def main():
    """main method"""
    test_set_up()


if __name__ == "__main__":
    main()