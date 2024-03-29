# sentences-vector-search

## Инструкция

Самый быстрый способ посмотреть на работу кода - запустить единственную ячейку в блокноте Google Colab (нужно подключиться с GPU, без GPU будет очень долго):

https://colab.research.google.com/drive/1f8m-ROem4SQmjX3xMDBYFIsDZIa3T2sH?usp=sharing

При запуске на локальном компьютере могут возникнуть проблемы с зависимостями.

### 1. Создание корпуса предложений

В файле **article_names.txt** указываем названия всех статей из Википедии, которые нужно прочитать, каждое с новой строки.

В файле **skip_sections.txt** указываем названия разделов статей, которые не нужно обрабатывать, каждое с новой строки.

В файле **user_agent.txt** указываем строку ```User-Agent```, которую просят отправлять вместе с запросами авторы API Википедии.
Формат следующий: ```ProjectName (email@server.domain)```

Запускаем скрипт **sentences.py** из командной строки, например:
```
$ python sentences.py
```
Скрипт выполняется в течение какого-то времени (между запросами к Википедии паузы по 0.3 секунды, чтобы не слишком нагрузить сервер). После того, как скрипт выполнится, в одной папке с ним появится файл **corpus.pkl**, содержащий list of str с найденными предложениями.

### 2. Векторизация корпуса предложений

Помещаем файл **corpus.pkl** в одну папку со скриптом **vectorize.py**, если они почему-то в разных папках.

Запускаем скрипт **vectorize.py** из командной строки, например:
```
$ python vectorize.py
```
Скрипт выполняется в течение какого-то времени (он качает модель и токенайзер, да и сам процесс векторизации занимает время). На CPU векторизация 2000 предложений занимает порядка 40 минут, на GPU - около 1 минуты.
После того, как скрипт выполнится, в одной папке с ним появится файл **vector_passages.pkl**, содержащий numpy.ndarray of float32 с нормализованными эмбеддингами предложений.

### 3. Поиск по векторизованным предложениям

Убеждаемся, что файлы **vector_passages.pkl** и **corpus.pkl** лежат в одной папке со скриптом **vectorize.py**. 
```
>>> import vectorize
>>> results, scores = vectorize.search('Самый большой стадион для игры в крикет вмещает 132 000 зрителей.', n=2)
>>> for score, result in zip(scores, results):
>>>     print('{:.2f}'.format(score), result)
>>>
0.93 Это крупнейший стадион для крикета в мире, который вмещает 132 тысячи зрителей.
0.83 Турнир является одним из наиболее известных состязаний по крикету и поныне.
```
