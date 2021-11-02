import pytest
import sys

sys.path.append("C:/Users/dakin/Documents/Personal Projects/Opinion_Detection")
from get_tweets import *


def test_set_up():
    (api, connection) = set_up("../twitter_token.json.txt")
    user = api.get_user(screen_name="twitter")
    user_name = user.screen_name()
    assert user_name == "Twitter"
    
    
