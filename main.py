#1. raw_data.csv - Dữ liệu thô
#2. processed_data.csv - Dữ liệu đã xử lý
#3. news_with_topics.csv - Kết quả phân tích chủ đề
#4. topics_summary.csv - Tóm tắt các chủ đề
#5. eda_word_distribution.png - Biểu đồ phân tích mô tả
#6. topic_analysis_results.png - Biểu đồ kết quả phân tích
#7. topic_wordclouds.png - Word clouds các chủ đề


import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import time
from datetime import datetime
import warnings
from matplotlib.pyplot import title

warnings.filterwarnings('ignore')

# Trực quan hóa dữ liệu
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from underthesea import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans


#1. THU THẬP DỮ LIỆU


class NewsCollector:

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.articles = []

    def collect_vnexpress(self, max_articles):
        print("\n1. Đang thu thập tin từ VNExpress\n")
        categories = [
            'https://vnexpress.net/thoi-su',
            'https://vnexpress.net/kinh-doanh',
            'https://vnexpress.net/giai-tri',
            'https://vnexpress.net/the-thao',
            'https://vnexpress.net/khoa-hoc-cong-nghe',
            'https://vnexpress.net/bat-dong-san',
            'https://vnexpress.net/suc-khoe',
            'https://vnexpress.net/giao-duc',
            'https://vnexpress.net/doi-song',
            'https://vnexpress.net/oto-xe-may',
            'https://vnexpress.net/du-lich',
        ]

        count = 0
        for category_url in categories:
            if count >= max_articles:
                break
            try:
                response = requests.get(category_url, headers=self.headers, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')

                articles = soup.find_all('article', class_='item-news', limit=30)

                for article in articles:
                    if count >= max_articles:
                        break

                    try:
                        title_tag = article.find('h3', class_='title-news')
                        desc_tag = article.find('p', class_='description')

                        if title_tag and desc_tag:
                            title = title_tag.get_text(strip=True)
                            description = desc_tag.get_text(strip=True)

                            self.articles.append({
                                'source': 'VNExpress',
                                'title': title,
                                'content': description,
                                'language': 'vi',
                                'collected_at': datetime.now()
                            })
                            count += 1
                    except Exception as e:
                        continue

                time.sleep(2)

            except Exception as e:
                print(f"Lỗi khi thu thập từ {category_url}: {e}")

        print(f"Đã thu thập {count} bài viết từ VNExpress")
        return count

    def collect_tuoitre(self, max_articles=300):
        print("\n3. Thu thập tin từ Tuổi Trẻ\n")

        categories = [
            'https://tuoitre.vn/thoi-su.htm',
            'https://tuoitre.vn/kinh-doanh.htm',
            'https://tuoitre.vn/cong-nghe.htm',
            'https://tuoitre.vn/giai-tri.htm',
            'https://tuoitre.vn/the-thao.htm',
            'https://tuoitre.vn/giao-duc.htm'
        ]

        cnt = 0
        for url in categories:

            if cnt >= max_articles:
                break

            try:

                current_headers = self.headers.copy()
                current_headers['Referer'] = 'https://tuoitre.vn/'
                response = requests.get(url, headers=self.headers, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')

                items = soup.find_all('div', class_='box-category-item')

                for item in items:
                    if cnt >= max_articles:
                        break

                    title_tag = item.find('a', class_='box-category-link-title')
                    desc_tag = item.find('p', class_='box-category-sapo')

                    if title_tag:
                        title = title_tag.get_text(strip=True)
                        description = desc_tag.get_text(strip=True) if desc_tag else title

                        self.articles.append({
                            'source': 'Tuoi Tre',
                            'title': title,
                            'content': description,
                            'language': 'vi',
                            'collected_at': datetime.now()
                        })
                        cnt += 1

                time.sleep(2)

            except Exception as e:
                print(f"Lỗi thu thập từ {url}: {e}")

        print(f"  Đã thu thập {cnt} bài viết từ Tuổi Trẻ")
        return cnt

    def collect_cnn(self, max_articles):
        print("\n2. Thu thập tin từ CNN\n")
        categories = [
            'https://edition.cnn.com/world',
            'https://edition.cnn.com/politics',
            'https://edition.cnn.com/business',
            'https://edition.cnn.com/health',
            'https://edition.cnn.com/entertainment',
            'https://edition.cnn.com/style',
            'https://edition.cnn.com/travel',
            'https://edition.cnn.com/science'
        ]

        cnt = 0
        for categories_url in categories:
            if cnt >= max_articles:
                break
            try:
                response = requests.get(categories_url, headers=self.headers, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                headlines = soup.find_all(['h2', 'h3'], limit=50)

                for headline in headlines:
                    if cnt >= max_articles:
                        break

                    try:
                        text = headline.get_text(strip=True)
                        if len(text) > 20:
                            self.articles.append({
                                'source': 'CNN',
                                'title': text,
                                'content': text,
                                'language': 'en',
                                'collected_at': datetime.now()
                            })
                            cnt += 1
                    except Exception as e:
                        continue

                time.sleep(2)

            except Exception as e:
                print(f"Lỗi thu thập từ {categories}: {e}")

        print(f"  Đã thu thập {cnt} bài viết từ CNN")
        return cnt

    def collect_vietnamnet(self, max_articles=300):
        print("\n4. Thu thập tin từ VietnamNet")
        categories = [
            'https://vietnamnet.vn/chinh-tri',
            'https://vietnamnet.vn/thoi-su',
            'https://vietnamnet.vn/kinh-doanh',
            'https://vietnamnet.vn/dan-toc-ton-giao',
            'https://vietnamnet.vn/giao-duc',
            'https://vietnamnet.vn/the-gioi',
            'https://vietnamnet.vn/van-hoa-giai-tri',
            'https://vietnamnet.vn/tuan-viet-nam'
        ]

        cnt = 0

        for categories_url in categories:
            if cnt > max_articles:
                break
            try:
                response = requests.get(categories_url, headers=self.headers, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                headlines = soup.find_all(['h2', 'h3'], limit=50)
                for headline in headlines:
                    if cnt >= max_articles:
                        break

                    try:
                        text = headline.get_text(strip=True)
                        if len(text) > 20:
                            self.articles.append({
                                'source': 'VietnamNet',
                                'title': text,
                                'content': text,
                                'language': 'vi',
                                'collected_at': datetime.now()
                            })
                            cnt += 1
                    except Exception as e:
                        continue

                time.sleep(2)

            except Exception as e:
                print(f"Lỗi thu thập từ {categories}: {e}")

        print(f"  Đã thu thập {cnt} bài viết từ VietnamNet")
        return cnt

    def get_dataframe(self):
        return pd.DataFrame(self.articles)



print("PHẦN 1: THU THẬP DỮ LIỆUu")
collector = NewsCollector()
collector.collect_vnexpress(max_articles=300)
collector.collect_cnn(max_articles=300)
collector.collect_tuoitre(max_articles=300)
collector.collect_vietnamnet(max_articles=300)

df = collector.get_dataframe()
print(f"TỔNG SỐ BÀI VIẾT THU THẬP: {len(df)}")

# Lưu dữ liệu thô chưa xử lý
df.to_csv('raw_data.csv', index=False, encoding='utf-8-sig')
print("\n Đã lưu dữ liệu thô vào: news_raw_data.csv")


#2. TIỀN XỬ LÝ DỮ LIỆU

print("PHẦN 2: TIỀN XỬ LÝ DỮ LIỆU")

# 2.1 Làm sạch dữ liệu
print("\n2.1. Làm sạch dữ liệu")
df_clean = df.copy()
df_clean.drop_duplicates(subset=['title'], inplace=True)
df_clean.dropna(subset=['content'], inplace=True)

print(f"Số bài sau khi loại bỏ trùng lặp: {len(df_clean)}")

# 2.2 Hàm làm sạch văn bản
print("\n2.2. Làm sạch văn bản")
def clean_text(text, language='vi'):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-ZÀ-ỹ\s]", "", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df_clean['clean_content'] = df_clean.apply(
    lambda row: clean_text(row['content'], row['language']), axis=1
)


# 2.3 Tách từ cho tiếng Việt
print("\n2.3. Tách từ tiếng Việt")

def tokenize_text(text, language):
    if language == 'vi':
        return word_tokenize(text, format="text")
    else:
        return text

df_clean['tokens'] = df_clean.apply(
    lambda row: tokenize_text(row['clean_content'], row['language']), axis=1
)

#2.4 Lưu dữ liệu đã xử lý
df_clean.to_csv('processed_data.csv', index=False, encoding='utf-8-sig')




#3. PHÂN TÍCH DỮ LIỆU MÔ TẢ
print("\nPHẦN 3: PHÂN TÍCH DỮ LIỆU MÔ TẢ")
#3.1 Thống kê cơ bản
print("\nThống kê theo nguồn tin:")
print(df_clean['source'].value_counts())


#3.2 Phân bố độ dài văn bản
df_clean['word_count'] = df_clean['tokens'].apply(lambda x: len(x.split()))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(data=df_clean, x='word_count', bins=30, kde=True)
plt.title('Phân bố độ dài bài viết')
plt.xlabel('Số từ')
plt.ylabel('Tần suất')

plt.subplot(1, 2, 2)
sns.boxplot(data=df_clean, x='source', y='word_count')
plt.title('Độ dài bài viết theo nguồn')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('eda_word_distribution.png', dpi=300, bbox_inches='tight')


#4. TF-IDF VECTORIZATION
print("\nPHẦN 4: TF-IDF VECTORIZATION")
#4.1 Tạo TF-IDF vectors
vietnamese_stopwords = ['của', 'và', 'có', 'trong', 'được', 'này', ...]
tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2),
    stop_words=vietnamese_stopwords
)

tfidf_matrix = tfidf_vectorizer.fit_transform(df_clean['tokens'])

#4.2 Top từ khóa
feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_sum = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
top_indices = tfidf_sum.argsort()[-20:][::-1]

print("\nTop 20 từ khóa (TF-IDF):")
for idx in top_indices:
    print(f"  - {feature_names[idx]}: {tfidf_sum[idx]:.2f}")


#5. TOPIC MODELING VỚI LDA
print("PHẦN 5: TOPIC MODELING VỚI LDA")
 

#5.1 Chuẩn bị dữ liệu cho LDA
count_vectorizer = CountVectorizer(
    max_features=1000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2),
    stop_words='english'
)

count_matrix = count_vectorizer.fit_transform(df_clean['tokens'])

#5.2 Huấn luyện mô hình LDA
lda_model = LatentDirichletAllocation(
    n_components= 5,
    random_state=42,
    max_iter=20,
    learning_method='online',
    n_jobs=-1
)

lda_output = lda_model.fit_transform(count_matrix)
print("Huấn luyện mô hình LDA với 5 chủ đề")


#5.3 Hiển thị các chủ đề
print("CÁC CHỦ ĐỀ PHÁT HIỆN ĐƯỢC BẰNG MÔ HÌNH LDA: ")
feature_names = count_vectorizer.get_feature_names_out()


def display_topics(model, feature_names, n_top_words=10):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        topics.append(top_words)

        print(f"\nChủ đề {topic_idx + 1}:")
        print(f"  Từ khóa: {', '.join(top_words)}")

    return topics

topics = display_topics(lda_model, feature_names, n_top_words=10)

#5.4 Gán chủ đề cho mỗi bài viết
df_clean['dominant_topic'] = lda_output.argmax(axis=1)
df_clean['topic_probability'] = lda_output.max(axis=1)



#6. PHÂN CỤM VỚI K-MEANS
print("PHẦN 6: PHÂN CỤM VỚI K-MEANS")

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df_clean['kmeans_cluster'] = kmeans.fit_predict(tfidf_matrix)



#7. TRỰC QUAN HÓA KẾT QUẢ THU THẬP ĐƯỢC
print("PHẦN 7: TRỰC QUAN HÓA KẾT QUẢ")

# 7.1 Phân bố theo chủ đề
fig, axes = plt.subplots(2, 2, figsize=(15, 12))


#Biểu đồ 1: Phân bố chủ đề LDA
sns.countplot(data=df_clean, x='dominant_topic', ax=axes[0, 0], palette='viridis')
axes[0, 0].set_title('Phân bố các chủ đề (LDA)', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Chủ đề')
axes[0, 0].set_ylabel('Số bài viết')



#Biểu đồ 2: Phân bố cụm K-Means
sns.countplot(data=df_clean, x='kmeans_cluster', ax=axes[0, 1], palette='coolwarm')
axes[0, 1].set_title('Phân bố các cụm (K-Means)', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('Cụm')
axes[0, 1].set_ylabel('Số bài viết')



#Biểu đồ 3: Độ tin cậy chủ đề
sns.histplot(data=df_clean, x='topic_probability', bins=30, kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Phân bố độ tin cậy chủ đề', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Xác suất')
axes[1, 0].set_ylabel('Tần suất')



#Biểu đồ 4: Chủ đề theo nguồn
pd.crosstab(df_clean['source'], df_clean['dominant_topic']).plot(
    kind='bar', stacked=True, ax=axes[1, 1], colormap='Set3'
)
axes[1, 1].set_title('Chủ đề theo nguồn tin', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Nguồn tin')
axes[1, 1].set_ylabel('Số bài viết')
axes[1, 1].legend(title='Chủ đề', bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.savefig('topic_analysis_results.png', dpi=300, bbox_inches='tight')


#7.2 Word Cloud cho mỗi chủ đề thu thập được
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for topic_idx in range( 5):
    topic_docs = df_clean[df_clean['dominant_topic'] == topic_idx]['tokens']
    topic_text = ' '.join(topic_docs)

#7.2.1 tạo wordcloud
    wordcloud = WordCloud(
        width=480,
        height=360,
        background_color='white',
        colormap='viridis',
        max_words=50
    ).generate(topic_text)

    axes[topic_idx].imshow(wordcloud, interpolation='bilinear')
    axes[topic_idx].set_title(f'Chủ đề {topic_idx + 1}', fontsize=13, fontweight='bold')
    axes[topic_idx].axis('off')

axes[-1].axis('off')

plt.tight_layout()
plt.savefig('topic_wordclouds.png', dpi=360, bbox_inches='tight')



#8. LƯU KẾT QUẢ CUỐI CÙNG
print("LƯU KẾT QUẢ")

df_final = df_clean[['source', 'title', 'content', 'dominant_topic', 'topic_probability', 'kmeans_cluster', 'word_count']]
df_final.to_csv('news_with_topics.csv', index=False, encoding='utf-8-sig')

topics_df = pd.DataFrame({
    'Topic': [f'Topic {i + 1}' for i in range( 5)],
    'Keywords': [', '.join(topic) for topic in topics],
    'Article_Count': [len(df_clean[df_clean['dominant_topic'] == i]) for i in range( 5)]
})
topics_df.to_csv('topics_summary.csv', index=False, encoding='utf-8-sig')