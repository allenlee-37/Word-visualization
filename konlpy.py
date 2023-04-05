import pandas as pd
import re
import nltk
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
from PIL import Image

''' 출력 단어 갯수 '''
word_volume = 50

''' 워드클라우에서 삭제할 단어 추가 '''
custom_stopwords = ['example', 'of', 'stop', 'words', '``',
                    'wa', 'ha', 'aslo',
                    ]

def load_data(fileName): return pd.read_excel(f'./data/{fileName}', index_col=0)

def text_cleaning(text):
    # text = re.sub(r'[^.,?!\s\w]',' ',str(text)) # 특수문자 삭제
    text = re.sub(r"[.“`’,``?”!'₩`:()]",' ',str(text)) # 특수문자 삭제
    text = re.sub(r'[0-9]','',str(text)) # 특수문자 삭제
    text = text.lower()
    return text

def tokenization(text_list):
    texts = ''.join(text_list)
    # 토큰화
    word_tokens = nltk.word_tokenize(texts)
    # POS tagging
    return nltk.pos_tag(word_tokens)

def lemmatize(tokens_pos):
    # Lemmatization (표제어 추출)
    wlem = nltk.WordNetLemmatizer()
    return [wlem.lemmatize(word) for word, pos in tokens_pos]

def remove_stopwords(unique_words):
    stopwords_list = stopwords.words('english') # nltk에서 제공하는 불용어사전 이용
    result = unique_words.copy()
    for word in unique_words:
        if word in stopwords_list:
            result.remove(word)
    return result

def remove_custom_stopwords(word_dict, custom_stopwords=custom_stopwords):
    result = word_dict.copy()
    for word in word_dict:
        if word in custom_stopwords:
            result.remove(word)
    return result

def count_words(lemmatized_words, word_volume):
    c = Counter(lemmatized_words)
    common_words_tuple = c.most_common(word_volume)
    # common_words_dict = dict((x,y) for x, y in common_words_tuple)
    common_words_dict = dict(common_words_tuple)
    return common_words_dict

def make_wordcloud(words):
    cand_mask=np.array(Image.open('./data/circle2.png'))
    wordcloud = WordCloud(
        font_path = 'NanumGothicBold.ttf', # 한글 글씨체 설정
        background_color='white', # 배경색은 흰색으로 
        colormap='Reds', # 글씨색은 빨간색으로
        mask=cand_mask, # 워드클라우드 모양 설정
    ).generate_from_frequencies(words)

    # 사이즈 설정 및 출력
    plt.figure(figsize=(50,50))
    plt.imshow(wordcloud,interpolation='bilinear')
    plt.axis('off') # 차트로 나오지 않게
    plt.savefig(f"./result/test.png")
    return

def main():
    # 데이터 로드
    df = load_data('korea medical-2023-04-04-result.xlsx')
    # 텍스트 전처리
    df['title'] = list(map(lambda x: text_cleaning(x), df['title']))
    df['body'] = list(map(lambda x: text_cleaning(x), df['body']))
    # 단어 토큰화
    token_pos = tokenization(df['body'])
    # 표제어 추출
    lemmatized_words = lemmatize(token_pos)
    # 중복 X 단어
    unique_words = list(set(lemmatized_words))
    
    # 표제어에서 불필요 단어 삭제
    processed_lemmatized_words = remove_stopwords(lemmatized_words)
    processed_lemmatized_words = remove_custom_stopwords(processed_lemmatized_words)
    # 동일하게 단어 사전에서도 불필요 단어 삭제
    word_dict = remove_stopwords(unique_words)
    word_dict = remove_custom_stopwords(word_dict)
    word_dict.sort()

    common_words = count_words(processed_lemmatized_words, word_volume)
    make_wordcloud(common_words)

if __name__ == "__main__":
    main()