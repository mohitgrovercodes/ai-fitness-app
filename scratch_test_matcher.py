import sys
import os

# Ensure the root directory is in the python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.utils.gif_utils import media_matcher

print(media_matcher.get_media("Bird-Dog (Quadruped)"))
print(media_matcher.get_media("Glute Bridge (Bodyweight)"))
