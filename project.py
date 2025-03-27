from sklearn.preprocessing import MultiLabelBinarizer
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import streamlit as st 
import pandas as pd
# Функція для оцінки рекомендацій на основі події
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Лематизація
# stanza.download('uk')
# nlp = stanza.Pipeline('uk')
# def lemmatize_text_stanza(text):
#     if isinstance(text, str):
#         doc = nlp(text)
#         lemmatized_words = []
#         # Обробка по реченнях
#         for sentence in doc.sentences:
#             lemmatized_sentence = [word.lemma for word in sentence.words]
#             lemmatized_words.extend(lemmatized_sentence)
#         return " ".join(lemmatized_words)
#     else:
#         # Якщо не рядок, просто повертаємо текст без змін
#         return text
# Завантаження даних
def load_events(doc_path):
    events_df = pd.read_csv(doc_path)
    events = {}
    # Перебираємо рядки таблиці
    for _, row in events_df.iterrows():
        event = row['Подія']  # Зберігаємо назву події
        items = row.dropna().values[1:]  # Вибираємо всі товари (крім назви події)
        events[event] = list(items)  # Додаємо в словник
    return events
# Завантажуємо дані з Excel (товари в магазинах)
def load_products(excel_path):
    df = pd.read_excel(excel_path, sheet_name=None)
    return df
def remove_digits_and_punctuation(text):
    if isinstance(text, str):  # Перевірка, чи є значення рядком
        # Спочатку видаляємо цифри
        text = re.sub(r'\d+', '', text)  # Видаляє всі цифри
        
        # Потім замінюємо всі спеціальні символи (крім апострофа) на пробіл
        text = ''.join([char if not (char in string.punctuation and char != "'") else ' ' for char in text])
        
    return text  # Повертаємо оновлений текст
# функція обробки даних
# def edit_frame(df):
#     all_product_descriptions = []  
#     for store in df.keys():
#         store_data = df[store]
#         df[store] = store_data
#         # Перевірка, чи є потрібні стовпці для створення опису
#         if 'Категорія' in store_data.columns and 'Назва товару' in store_data.columns:
#             # Створюємо опис для товару (пов'язуємо категорію, назву товару і рівень категорії)
#             store_data['Description'] = store_data['Категорія'].fillna('') + " " + store_data['Назва товару'].fillna('') + " " + store_data['Рівень категорії 3'].fillna('')
#             store_data['Description'] = store_data['Description'].fillna('')  # Заповнюємо порожні значення
#             store_data['Description'] = store_data['Description'].apply(lambda x: x.lower())
#             store_data['Description'] = store_data['Description'].apply(remove_digits_and_punctuation)
#             store_data['Description'] = store_data['Description'].apply(lemmatize_text_stanza)
#             all_product_descriptions.extend(store_data['Description'].fillna(''))  # Збираємо всі описи товарів
#         else:
#             print(f"Стовпці 'Категорія' або 'Назва товару' відсутні для магазину {store}")
#     return df, all_product_descriptions
def vectorize_products_and_events(all_product_descriptions, events,stop_words):
    # Створюємо TfidfVectorizer для товарів та подій
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    # Векторизація товарів
    product_matrix = vectorizer.fit_transform(all_product_descriptions)
    # Векторизація описів подій
    event_descriptions = []
    for event, products in events.items():
        event_desc = " ".join(products)
        event_descriptions.append(event_desc)
    if len(event_descriptions) == 0:
        raise ValueError("Описів подій не знайдено для векторизації")
    event_matrix = vectorizer.transform(event_descriptions)
    return product_matrix, event_matrix, vectorizer
# Функція для рекомендацій на основі події
def recommend_for_event(event_name, events, df, product_matrix, event_vectorizer):
    # Векторизуємо опис події
    event_description = " ".join(events.get(event_name, []))
    # Векторизація опису події за допомогою векторизатора (не event_matrix)
    event_vector = event_vectorizer.transform([event_description])  # Тепер використовуємо event_vectorizer
    # Обчислюємо косинусну схожість між описом події та товарами в магазині
    cosine_sim = cosine_similarity(event_vector, product_matrix)
    # Отримуємо індекси найбільш схожих товарів
    most_similar_products = cosine_sim.argsort()[0][-10:]
    recommended_products = []
    # Додаємо рекомендовані товари
    recommended_products = [df.iloc[index]['Product Name'] for index in most_similar_products]
    recommended_products = list(set(recommended_products))
    return recommended_products
def recommend_based_on_cheques(product_name, cosine_sim, top_n):
        product_df['Product Name'] = product_df['Product Name'].apply(clean_product_name)
        if product_name in product_df['Product Name'].values:
            product_idx = product_df[product_df['Product Name'] == product_name].index[0]
            # Перевірка, чи є індекс в межах матриці cosine_sim
            if product_idx < cosine_sim.shape[0]:
                sim_scores = list(enumerate(cosine_sim[product_idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                recommended_idx = [x[0] for x in sim_scores[1:top_n+1]]
                recommended_based_on_cheques = product_df.iloc[recommended_idx]['Product Name'].tolist()
                return recommended_based_on_cheques
            else:
                return []
        else:
            print(f"Товар {product_name} не знайдено в базі даних.")
            return []

def generate_stop_words_from_file(all_product_descriptions, additional_stop_words=None, max_word_length=3):
    # Векторизація товарів
    vectorizer = TfidfVectorizer(stop_words='english', max_features=300)  # Обмежуємо кількість слів
    tfidf_matrix = vectorizer.fit_transform(all_product_descriptions)
    # Отримуємо список слів (фіч)
    features = vectorizer.get_feature_names_out()
    # Якщо додаткові стоп-слова не задані, ініціалізуємо порожнім списком
    if additional_stop_words is None:
        additional_stop_words = []
    filtered_stop_words = [word for word in features if len(word) <= max_word_length]
    # Поєднуємо найбільш поширені слова та додаткові стоп-слова
    stop_words = set(filtered_stop_words).union(set(additional_stop_words))
    return stop_words
def clean_stop_words(stop_words):
    # Очищаємо список від пустих рядків і перевіряємо, чи є всі елементи рядками
    unwanted_words = ['об', 'см', 'мл', 'на', 'по', 'за', 'це', 'жін', 'чол','зим','чаю', 'лак','вій', 'usb', 'кух','шв', 'зуб','men','еко','шок', 'мит', 'авт', 'чищ', 'aux','led', 'туш', 'сад','чай','гра','шию','дит','укр','рук']
    stop_words = [word for word in stop_words if word not in unwanted_words]
    return [word for word in stop_words if isinstance(word, str) and len(word) > 1]
def clean_product_name(product_name):
    # Видаляємо всі цифри та пробіли перед словом
    cleaned_name = re.sub(r'^\s*\d+\s*', '', product_name)  # Видаляє числа і пробіли на початку рядка
    return cleaned_name.strip()  # Видаляє пробіли на початку та в кінці
def create_purchase_matrix(df_cheques):
    mlb = MultiLabelBinarizer()
    matrix = mlb.fit_transform(df_cheques.groupby('Номер чеку')['Назва товару'].apply(list))
    # Перетворюємо результат у DataFrame для зручності
    purchase_matrix = pd.DataFrame(matrix, columns=mlb.classes_)
    return purchase_matrix
def normalize_text(text):
    text = text.lower()  # Перетворення на малий регістр
    text = re.sub(r'[^\w\s]', '', text)  # Видалення пунктуації
    return text

# Функція для пошуку схожих подій
def find_similar_events(input_event, events, top_n=3):
    """
    Функція для знаходження найсхожіших подій до введеного запиту.
    
    :param input_event: введена подія (стрічка)
    :param events: словник подій, де ключ — назва події, значення — список товарів
    :param top_n: кількість найсхожіших подій для виведення
    :return: список найсхожіших подій
    """
    # Перетворюємо події в текстові описи
    event_descriptions = {event: " ".join(items) for event, items in events.items()}
    # Токенізація та векторизація
    vectorizer = TfidfVectorizer()
    event_matrix = vectorizer.fit_transform(event_descriptions.values())
    # Нормалізуємо введену подію
    input_event_normalized = normalize_text(input_event)
    # Векторизуємо введену подію
    input_event_vector = vectorizer.transform([input_event_normalized])
    # Обчислюємо косинусну схожість
    cosine_sim = cosine_similarity(input_event_vector, event_matrix)
    # Отримуємо найбільш схожі події
    most_similar_events_indices = cosine_sim.argsort()[0][-top_n:]  # Топ N найбільш схожих
    # Повертаємо схожі події
    similar_events = [(list(event_descriptions.keys())[idx])  for idx in most_similar_events_indices]
    return similar_events

# Функція для векторизації товарів
def vectorize_products_and_events(all_product_descriptions, events, stop_words):
    # Створюємо TfidfVectorizer для товарів та подій
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    # Векторизація товарів (для магазину)
    product_matrix = vectorizer.fit_transform(all_product_descriptions)
    # Векторизація описів товарів для подій
    event_descriptions = []
    for event, products in events.items():
        event_desc = " ".join(products)
        event_descriptions.append(event_desc)
    event_matrix = vectorizer.transform(event_descriptions)
    return product_matrix, event_matrix, vectorizer

# Завантажуємо два файли чеків
file_path_1 = "/Users/macbook/Desktop/avrora/чеки1.csv"  # Вкажіть шлях до файлу
file_path_2 = "/Users/macbook/Desktop/avrora/чеки2.csv"  # Вкажіть шлях до файлу

df1 = pd.read_csv(file_path_1)
df2 = pd.read_csv(file_path_2)
df_cheques = pd.concat([df1, df2], ignore_index=True)
# Шлях до файлів
doc_path = "/Users/macbook/Desktop/avrora/events_lemmatized.csv"
excel_path = "/Users/macbook/Desktop/avrora/[1] - Категорії_товари_залишки_вартість.xlsx"

# Завантажуємо дані
events = load_events(doc_path)
product_df = pd.read_csv("/Users/macbook/Desktop/avrora/products_data.csv")
product_df['Product Name'] = product_df['Product Name'].apply(clean_product_name)
def main():
    print("Рекомендаційна система для товарів та подій")
    # Завантажуємо два файли чеків
    file_path_1 = "/Users/macbook/Desktop/avrora/чеки1.csv"  # Вкажіть шлях до файлу
    file_path_2 = "/Users/macbook/Desktop/avrora/чеки2.csv" 

    df1 = pd.read_csv(file_path_1)
    df2 = pd.read_csv(file_path_2)
    df_cheques = pd.concat([df1, df2], ignore_index=True)
    doc_path = "/Users/macbook/Desktop/avrora/events_lemmatized.csv"
    # Завантажуємо дані
    events = load_events(doc_path)
    product_df = pd.read_csv("/Users/macbook/Desktop/avrora/products_data.csv")
    product_df['Product Name'] = product_df['Product Name'].apply(clean_product_name)
    # df, all_product_descriptions = edit_frame(df)
    all_product_descriptions = []
    all_product_descriptions.extend(product_df['Description'].fillna(''))  # Збираємо всі описи товарів
    # Генеруємо стоп-слова
    additional_stop_words = ["д", "арт", 'асорт', 'в', 'з', 'набір', 'gf']
    stop_words = generate_stop_words_from_file(all_product_descriptions, additional_stop_words)
    stop_words = clean_stop_words(stop_words)
    purchase_matrix = create_purchase_matrix(df_cheques)
    cheques_sim = cosine_similarity(purchase_matrix.T) 
    product_matrix, event_matrix, event_vectorizer = vectorize_products_and_events(all_product_descriptions, events, stop_words)

    event_name = input("Введіть назву події:")
    if event_name in events.keys():
        recommended_products = recommend_for_event(event_name, events, product_df, product_matrix, event_vectorizer)       
        recommended_products_str = ", ".join(recommended_products)       
        print(f"\nРекомендовані товари для події '{event_name}': {recommended_products_str}")
        for product in recommended_products:
            product_name = product
            recommended_based_on_cheques = recommend_based_on_cheques(product_name, cheques_sim, top_n=1)
            if len(recommended_based_on_cheques) != 0:
                # Перетворюємо рекомендовані товари на рядок
                recommended_based_on_cheques_str = ", ".join(recommended_based_on_cheques)
                if product == recommended_products[0]:
                    print("З цим часто купують:\n")
                print(f"- {recommended_based_on_cheques_str}")
    else:
        similar_events = find_similar_events(event_name, events)
        print(f"\nРекомендовані товари для схожих до '{event_name}' подій:")
        if len(similar_events) != 0:
            for event in similar_events:
                event_name = str(event)
                recommended_products = recommend_for_event(event_name, events, product_df, product_matrix, event_vectorizer)
                # Конвертуємо список рекомендованих товарів для схожих подій у рядок
                recommended_products_str = ", ".join(recommended_products)
                print(f"\t'{event_name}': {recommended_products_str}")
                for product in recommended_products:
                    product_name = product
                    recommended_based_on_cheques = recommend_based_on_cheques(product_name, cheques_sim, top_n=1)
                    if len(recommended_based_on_cheques) != 0:
                        # Перетворюємо рекомендовані товари на рядок
                        recommended_based_on_cheques_str = ", ".join(recommended_based_on_cheques)
                        if product == recommended_products[0]:
                            print("\tЗ цим часто купують:")
                        print(f"\n- {recommended_based_on_cheques_str}»")
if __name__ == "__main__":
    main()