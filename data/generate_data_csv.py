import pandas as pd
import numpy as np
import pickle as pkl
from collections import defaultdict
import copy
from collections import Counter


# get txt file
def write_seq(User, dataset_name):
    f = open(f'{dataset_name}/{dataset_name}.txt','w')
    for user in User.keys():
        f.write('%d '%(user))
        for interaction in User[user]:
            f.write('%d ' %(interaction[0])) # extract  itemid
        f.write('\n')

        
# item purchase period 
# def calculate_purchase_intervals(interactions):
#     user_items = defaultdict(list)

#     for user, interactions_list in interactions.items():
#         item_last_timestamp = {} 

#         for itemid, timestamp_list in interactions_list:
#             timestamp = timestamp_list[2]
#             if itemid in item_last_timestamp:
#                 time_diff = abs(timestamp - item_last_timestamp[itemid])
#                 user_items[user].append([itemid, timestamp_list,time_diff]) 
#             else:
#                 user_items[user].append([itemid, timestamp_list,0])

#             item_last_timestamp[itemid] = timestamp

#     return user_items

def factorize_itemid(itemid_list):
    # 딕셔너리 초기화 (PAD 값을 0으로 설정)
    id_to_factorized = {'PAD': 0}
    next_id = 1  # 다음 factorized id는 1부터 시작

    for itemid in itemid_list:
        if itemid not in id_to_factorized:
            # 새로운 itemid를 factorized id로 매핑
            id_to_factorized[itemid] = next_id
            next_id += 1  # 다음 factorized id 증가

    return id_to_factorized
# item period calculate

def calculate_purchase_intervals(interactions):
    item_last_timestamp = {}
    item_timestamp_diffs = defaultdict(list)  # 각 아이템의 시간 간격을 저장할 딕셔너리
    interactions_copy = copy.deepcopy(interactions)
    # 각 사용자별로 상호작용 아이템 순회
    for user, interactions_list in interactions_copy.items():
        for itemid, timestamp_list in interactions_list:
            timestamp = timestamp_list[2]  # timestamp[2]는 day를 나타냄
            
            if itemid in item_last_timestamp:
                time_diff = abs(timestamp - item_last_timestamp[itemid])
                item_timestamp_diffs[itemid].append(time_diff)
            else:
                # 아이템에 대한 첫 번째 상호작용인 경우, 시간 간격을 0으로 설정
                item_timestamp_diffs[itemid].append(0)

            item_last_timestamp[itemid] = timestamp

    # 각 아이템에 대해 평균 시간 간격 계산 및 결과 저장
    for user, interactions_list in interactions_copy.items():
        updated_interactions_list = []
        for itemid, timestamp_list in interactions_list:
            if itemid in item_timestamp_diffs and len(item_timestamp_diffs[itemid]) >= 2:
                avg_time_diff = sum(item_timestamp_diffs[itemid]) / len(item_timestamp_diffs[itemid])
                timestamp_list_with_period = timestamp_list + [avg_time_diff]  # 주기를 timestamp_list 뒤에 추가
                updated_interactions_list.append([itemid, timestamp_list_with_period])
            else:
                timestamp_list_with_period = timestamp_list + [0]
                updated_interactions_list.append([itemid, timestamp_list_with_period])

        interactions_copy[user] = updated_interactions_list  # 업데이트된 interactions 리스트를 다시 저장

    return interactions_copy
def count_max_occurrence(row):
    counts = Counter(row)
    if counts:
        return max(counts.values())  # 가장 많이 등장한 아이템의 등장 횟수 반환
    else:
        return 0  # 빈 리스트인 경우 등장 횟수는 0



def generate_data(dataset_name, reviews_path):

    df= pd.read_csv(reviews_path)
    df = df[['user_id', 'product_id', 'brand', 'category_id', 'event_time']]
    df['timestamp']= pd.to_datetime(df['event_time']).astype(int) // 10**9
    

    # 5번 이상 리뷰 남긴 user, 5번 이상 리뷰 남겨진 item
    item_counts = df['product_id'].value_counts()
    selected_items = item_counts[item_counts >= 5].index.tolist()
    df=df[df['product_id'].isin(selected_items)]

    gropuby_df= df.groupby('user_id')['product_id'].apply(list).reset_index()
    gropuby_df['list_length'] = gropuby_df['product_id'].apply(len)
    selected_users = gropuby_df[gropuby_df['list_length'] >= 5]['user_id'].tolist()
    df = df[df['user_id'].isin(selected_users)]

    # sample 데이터 비율 조정 (각 user별 재구매 횟수 비율이 30% 미만만 남게 user 삭제)

    # gropuby_df['max_occurrence'] = gropuby_df['product_id'].apply(count_max_occurrence) # 각 user 별 최대 반복 구매 횟수
    # gropuby_df['ratio_product_length_max_repurchase']=gropuby_df['max_occurrence']/gropuby_df['product_id'].apply(lambda x:len(x))*100 # 비율 저장
    # ratio_selected_users=gropuby_df[gropuby_df['ratio_product_length_max_repurchase']<30]['user_id']
    # df=df[df['user_id'].isin(ratio_selected_users)]
    ##

    # ## sample (최대 2번 같은 아이템을 반복해서 구매한 user만 추출)
    # gropuby_df['max_occurrence'] = gropuby_df['product_id'].apply(count_max_occurrence) # 각 user 별 최대로 아이템을 반복해서 구매한 횟수
    # selected_users_repeat=gropuby_df[gropuby_df['max_occurrence']<=2]['user_id']
    # df=df[df['user_id'].isin(selected_users_repeat)]  
    # ##

    
    user2id={'[PAD]':0}
    item2id={'[PAD]':0}
    brand2id = {'[PAD]': 0} 
    category2id = {'[PAD]': 0}

    items_map={
        'item2price':{},
        'item2category': {},
        'item2brand': {}
    }
    user2id=factorize_itemid(df['user_id'])
    item2id=factorize_itemid(df['product_id'])
    brand2id=factorize_itemid(df['brand'])
    category2id=factorize_itemid(df['category_id'])

    # # item, price matching
    # result_dict_p = df.set_index('product_id')['price'].to_dict()
    # items_map['item2price'] ={str(key): '' if pd.isna(value) else value for key, value in result_dict_p.items()}
    
    # item, category matching
    result_dict_c= df.set_index('product_id')['category_id'].to_dict()
    items_map['item2category'] ={key: '' if pd.isna(value) else value for key, value in result_dict_c.items()}
    
    # item, brand matching
    result_dict_b= df.set_index('product_id')['brand'].to_dict()
    items_map['item2brand'] ={key: '' if pd.isna(value) else value for key, value in result_dict_b.items()}


    user_reviews = defaultdict(list)

    for index, row in df.iterrows():
        user_reviews[row['user_id']].append([row['product_id'],row['timestamp']])

    for user_id in user_reviews:
        user_reviews[user_id].sort(key=lambda x: x[1]) # timestamp별로 정렬

    item2category_id = defaultdict(list)
    categories_n_max=0

    for k in items_map['item2category'].keys():
        category= items_map['item2category'][k] # k:item, category:item's category
        if category2id[category] not in item2category_id[k]:
            item2category_id[k].append(category2id[category])
        categories_n_max = len(item2category_id[k]) if len(
            item2category_id[k]) > categories_n_max else categories_n_max
            
    for k in items_map['item2category'].keys():
        if items_map['item2category'][k] in category2id:
            item2category_id[k] = category2id[items_map['item2category'][k]]
        else:
            category2id[items_map['item2category'][k]] = len(category2id)
            item2category_id[k] = category2id[items_map['item2category'][k]]
            
    item2brand_id = {}
    for k in items_map['item2brand'].keys():
        if items_map['item2brand'][k] in brand2id:
            item2brand_id[k] = brand2id[items_map['item2brand'][k]]
        else:
            brand2id[items_map['item2brand'][k]] = len(brand2id)
            item2brand_id[k] = brand2id[items_map['item2brand'][k]]

    item_features = {0: [0] * 2} ##
    for k in items_map['item2brand'].keys():
        # category_feature = item2category_id[k] + (categories_n_max - len(item2category_id[k])) * [0]
        item_feature = [item2category_id[k]] +[item2brand_id[k]]
        assert len(item_feature) == len(item_features[0])
        item_features[item2id[k]] = item_feature


    item_features = list(item_features.values())        


    min_year = pd.to_datetime(np.array(df['timestamp']).min(), unit='s').year
    max_year = pd.to_datetime(np.array(df['timestamp']).max(), unit='s').year

    User = defaultdict(list)
    for u in user_reviews.keys():
        for item, action_time in user_reviews[u]:
            act_datetime = pd.to_datetime(action_time, unit='s')
            year = (act_datetime.year - min_year) / (max_year - min_year)
            month = act_datetime.month / 12
            day = act_datetime.day / 31
            dayofweek = act_datetime.dayofweek / 7
            dayofyear = act_datetime.dayofyear / 365
            week = act_datetime.week / 4
            context = [year, month, day, dayofweek, dayofyear, week]
            User[user2id[u]].append([item2id[item], context])

    item_period = calculate_purchase_intervals(User)

    data = {
        'user_seq': user_reviews,
        'items_map': items_map,
        'user_seq_token': User,
        'items_feat': item_features,
        'user2id': user2id,
        'item2id': item2id,
        'category2id': category2id,
        'brand2id': brand2id,
        'max_categories_n': categories_n_max,
        'item_period': item_period
    }

    pkl.dump(data, open(f'{dataset_name}/{dataset_name}.dat', 'wb'))
    print(f'generate_data:{dataset_name}.dat has finished!')

    write_seq(User,'e_commerce_cosmetic')  # write seq txt file 


if __name__ == '__main__':
    for dataset_name in ['e_commerce_cosmetic']:
        generate_data(dataset_name, f'{dataset_name}/{dataset_name}.csv')