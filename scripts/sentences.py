import os
import sys
import time
import re
import io
import pickle
import wikipediaapi as wapi
from bs4 import BeautifulSoup
from razdel import sentenize

TEXT_FILES_ENCODING = 'utf-8-sig'

SCRIPT_DIR = os.path.dirname(__file__)
PKL_FILE_PATH = os.path.join(SCRIPT_DIR, 'corpus.pkl')
ARTICLES_FILE_PATH = os.path.join(SCRIPT_DIR, 'article_names.txt')
SKIP_SECTIONS_FILE_PATH = os.path.join(SCRIPT_DIR, 'skip_sections.txt')
USER_AGENT_FILE_PATH = os.path.join(SCRIPT_DIR, 'user_agent.txt')

def file_is_present(file_path):
    if (os.path.exists(file_path) and os.path.isfile(file_path)):
        return True
    return False

for file_path in (ARTICLES_FILE_PATH, SKIP_SECTIONS_FILE_PATH, USER_AGENT_FILE_PATH):
    assert file_is_present(file_path), 'Не найден файл ' + file_path

with io.open(ARTICLES_FILE_PATH, mode='r', encoding=TEXT_FILES_ENCODING) as fp:
    article_names = [s.strip() for s in fp.read().split('\n')]
article_names = [s for s in article_names if s]

with io.open(SKIP_SECTIONS_FILE_PATH, mode='r', encoding=TEXT_FILES_ENCODING) as fp:
    skip_sections = [s.strip() for s in fp.read().split('\n')]
skip_sections = [s for s in skip_sections if s]

with io.open(USER_AGENT_FILE_PATH, mode='r', encoding=TEXT_FILES_ENCODING) as fp:
    user_agent = fp.read().strip()

def join_sections_recursive(p_wiki):
    ret = []
    sections = p_wiki.sections
    if len(sections) == 0:
        ret.append(p_wiki.text)
    else:
        for section in sections:
            if section.title in skip_sections:
                continue
            ret.extend(join_sections_recursive(section))
    return ret

wiki_html = wapi.Wikipedia(
    user_agent=user_agent,
    language='ru',
    extract_format=wapi.ExtractFormat.HTML
)

html_pages = []
for aname in article_names:
    try:
        page = wiki_html.page(aname)
        if page.exists():
            html_pages.append(page)
            time.sleep(0.3)
        else:
            print('Статья "' + aname.replace('_', ' ') + '" не найдена на Википедии.')
    except Exception as exc:
        print('Сервер Википедии недоступен.')
        break

soups = []
for page in html_pages:
    summary_text = page.summary + '\n'
    sections_texts = '\n'.join(join_sections_recursive(page))
    soup = BeautifulSoup(summary_text + sections_texts, features="html.parser")
    for item in soup.find_all('p'):
        item.append('\n')
    soups.append(soup)

corpus = []
for soup in soups:
    corpus.extend(list(x.text for x in sentenize(soup.get_text())))

for i, sentence in enumerate(corpus):
    corpus[i] = sentence.replace('\xa0', ' ')

class CleanTools:
    def __init__(self):
        self.lett_re = re.compile(r'[a-zA-Zа-я|ёА-Я|Ё]')
    def contains_letters(self, s):
        if self.lett_re.search(s):
            return True
        return False
    def check_brackets(self, s, opening='(', closing=')'):
        n = 0
        for c in s:
            if c == opening:
                n += 1
            elif c == closing:
                n -= 1
        return n
    def check_closing_bracket(self, s, opening='(', closing=')'):
        if closing not in s:
            return 0
        if opening not in s:
            return 1
        if s.index(closing) < s.index(opening):
            return 1
        return 0
    def brackets_ok(self, s, opening='(', closing=')'):
        if (
            self.check_brackets(s, opening=opening, closing=closing) == 0
            and
            self.check_closing_bracket(s, opening=opening, closing=closing) == 0
        ):
            return True
        return False

cleantools = CleanTools()

def clean_brackets(corpus, opening, closing):
    brackets_clean_corpus = []
    bad_indices = []

    for i, s in enumerate(corpus):
        if not cleantools.brackets_ok(s, opening=opening, closing=closing):
            bad_indices.append(i)

    cursor = 0
    while cursor < len(corpus):
        if cursor in bad_indices:
            if (cursor + 1) in bad_indices:
                two_sent = ' '.join(s.strip() for s in corpus[cursor:cursor+2])
                if cleantools.brackets_ok(two_sent, opening=opening, closing=closing):
                    brackets_clean_corpus.append(two_sent)
                    cursor += 2
                    continue
            if (cursor + 2) in bad_indices:
                three_sent = ' '.join(s.strip() for s in corpus[cursor:cursor+3])
                if cleantools.brackets_ok(three_sent, opening=opening, closing=closing):
                    brackets_clean_corpus.append(three_sent)
                    cursor += 3
                    continue
            brackets_clean_corpus.append(corpus[cursor])
            cursor += 1
        else:
            brackets_clean_corpus.append(corpus[cursor])
            cursor += 1
    return brackets_clean_corpus

def remove_repeated_char(s, char):
    ret = s
    charchar = char + char
    while (charchar) in ret:
        ret = ret.replace(charchar, char)
    return ret

corpus = clean_brackets(corpus, '(', ')')
corpus = clean_brackets(corpus, '«', '»')

corpus = list(map(lambda x: remove_repeated_char(x, '\n'), corpus))
corpus = list(map(lambda x: remove_repeated_char(x, ' '), corpus))
corpus = list(map(str.strip, corpus))

corpus = [s for s in corpus if cleantools.contains_letters(s)]

with io.open(PKL_FILE_PATH, mode='wb') as fp:
    pickle.dump(corpus, fp)

print('Количество предложений в корпусе: {}'.format(len(corpus)))
print('Корпус предложений записан в файл {}'.format(PKL_FILE_PATH))
