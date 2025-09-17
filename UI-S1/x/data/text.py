import re

def parse_tags(xml_content, tag_names):
    result = {}
    
    for tag_name in tag_names:
        # Define a regex pattern to match content for the current tag
        pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
        
        # Use re.search to find the first match of pattern in xml_content
        match = re.search(pattern, xml_content, re.DOTALL)
        
        if match:
            # Extract and return the captured content within the tags
            tag_content = match.group(1).strip()
            result[tag_name] = tag_content
        else:
            result[tag_name] = None
    
    return result

# def detect_repeat(long_string, repeat_threshold=5, length_threshold=10):
#     # 检查是否存在一个子串由长度大于length_threshold且重复repeat_threshold次构成
#     punctuation = r'!#%*+-./<=>@_|~'
#     pattern = rf'([^{punctuation}\s]{{{length_threshold},}})\1{{{repeat_threshold - 1}}}'
#     match = re.search(pattern, long_string)
#     if match:
#         return match.group(0)
#     else:
#         return None


def detect_repeat(long_string, repeat_threshold=5, length_threshold=10):
    # Adjust the pattern to include spaces in the allowed characters
    # to treat them as valid components of the repeat sequence.
    punctuation = r'!#%*+-./<=>@_|~'
    # Allows spaces within the repeated pattern
    pattern = rf'(([^{punctuation}]+?)\s?)\1{{{repeat_threshold - 1}}}' + ' ' * (repeat_threshold - 1)
    
    match = re.search(pattern, long_string)
    if match:
        return match.group(0)
    else:
        return None

if __name__ == '__main__':
    print(detect_repeat('----------------------------------------------------------------------'))
    print(detect_repeat('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'))
    print(detect_repeat("The narrative continues with the same animated person, still looking shocked and sweating, holding the smartphone. The text 'Jアラートが発令された...' (J-Alert was issued...) is reiterated. The scene transitions to another animated person with long black hair and a red shirt, also sweating and looking concerned in a domestic setting with a window showing the outside. The text reads '日本中が緊張感に包まれる中、' which translates to 'While the whole of Japan is enveloped in tension,'. The next frame shows another animated person with short black hair and a gray suit, sweating and looking worried in an office or public building. The text remains the same. A close-up emphasizes their worried expression. The scene then shifts to a screen displaying a series of text messages or forum posts, with the text 'ニュー速VIPではこの様なスレが立てられた' which translates to 'In News VIP, such threads were created'. The final frame of this clip shows an animated character with green hair and a green outfit, looking surprised with a hand over their mouth, accompanied by a cartoonish, elongated object resembling a missile. The text reads '僕の股間のミサイルも発射されそうです' which translates to 'It seems my crotch missile is about to launch'. This humorous twist adds a layer of unexpected comedy to the tense situation.", length_threshold=5))
    print(detect_repeat("The video continues to focus on the same dimly lit room with a greenish tint. The cluttered room and the white rocking chair remain unchanged, with the email address 'qloopnet@hotmail.com' still overlaid on the scene. The frames show minimal changes, highlighting the static nature of the room and the eerie stillness within it, with no visible movement or changes in the environment.",repeat_threshold=5, length_threshold=5))
    