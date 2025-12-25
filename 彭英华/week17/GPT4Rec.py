from rank_bm25 import BM25Okapi
from get_Query import Makedfdataset,predict,initialize_model_and_tokenizer
import itertools

# 网格搜索 BM25 参数 k1 和 b
def grid_search_bm25(movie_corpus, val_queries, val_relevant_movies, k1_values, b_values, K=5):
    best_score = 0
    best_params = (1.5,0.75)
    tokenized_corpus = [doc.lower().split() for doc in movie_corpus]

    for k1, b in itertools.product(k1_values, b_values):
        bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
        score_sum = 0.0
        for q, relevant in zip(val_queries, val_relevant_movies):
            q_tokens = q.lower().split()
            scores = bm25.get_scores(q_tokens)
            ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            retrieved_movies = [movie_corpus[idx].lower().strip() for idx in ranked_idx[:K]]
            # 计算召回率
            hits = len(set(retrieved_movies) & set([relevant.lower().strip()]))
            score_sum += hits
        avg_score = score_sum / len(val_queries)
        if avg_score > best_score:
            best_score = avg_score
            best_params = (k1, b)
    return best_params, best_score

# 用bm25进行检索
def retrieve_and_merge(queries, bm25,movie_titles,K=10):
    m = max(1, len(queries))
    per_q = K/m   #每个query需要检索的电影数
    final_list = []
    used = set()
  
    #queries 已经按得分顺序排列好
    for q in queries:
        q_tokens = q.lower().split()
        scores = bm25.get_scores(q_tokens)  #获取每个query对整个电影库的得分
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)  #得分进行排序
        
        # 对得分最高的Query先检索per_q个物品，然后将剩下的Query按照得分顺序分别去检索per_q个之前没有检索到的物品
        count = 0
        for idx in ranked_idx:
            if idx in used:
                continue
            final_list.append((movie_titles[idx], float(scores[idx])))
            used.add(idx)
            count += 1
            if count >= per_q or len(final_list) >= K:
                break
        if len(final_list) >= K:
            break
    return final_list[:K]

if __name__ =="__main__":
    train_df,_,test_df = Makedfdataset("./M_ML-100K/movies.dat",'./M_ML-100K/ratings.dat').getdataset() #获取训练集和测试集
    movie_titles = Makedfdataset("./M_ML-100K/movies.dat",'./M_ML-100K/ratings.dat').movie_titles   #获取所有电影名称
    movie_ids = list(range(len(movie_titles)))
    train_queries = []
    model_path = "./output_Qwen"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 获取训练集中每个样本生成的得分最高的Query
    train_prompts = train_df['prompt'].tolist()
    queries = predict(train_prompts, model, tokenizer, device="cuda", beam_size=5)
    for query in queries:
        train_queries.append(query[0])

    # 获取每个样本的原始电影名称
    train_relevant_movies = train_df['output'].tolist()

    # 进行表格搜索选择最优的k1和b
    k1_values = [0.5, 1.0, 1.5, 2.0]
    b_values = [0.3, 0.5, 0.75, 0.9]
    best_params, best_score = grid_search_bm25(movie_titles, train_queries, train_relevant_movies,
                                               k1_values, b_values, K=5)
    k1,b = best_params

    # 构建bm25的搜索库
    movie_corpus = [movie_title.lower().split() for movie_title in movie_titles]
    bm25 = BM25Okapi(movie_corpus, k1=k1, b=b)

    # 对测试集生成的Query进行检索top-10电影名称
    test_prompts = test_df['prompt'].tolist()
    test_queries = predict(test_prompts, model, tokenizer, device="cuda", beam_size=5)
    for query in test_queries:
        print(retrieve_and_merge(query,bm25,movie_titles,K=10))
