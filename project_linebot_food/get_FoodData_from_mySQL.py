#!/usr/bin/env python
# coding: utf-8

# In[72]:


import sql_acount
import random

def get_food_data():


    """
    登入mySQL，預設db = "for_foodgroup"
    """
    cursor=sql_acount.acount2()





    """
    隨機選擇3個R_id
    """
    R_id = [random.randrange(10000001, 10009999),random.randrange(10000001, 10009999),random.randrange(10000001, 10009999)]
    print(R_id)




    """
    從"for_foodgroup"資料庫中的"use_recipe_nutrients"表格，根據R_id，取得食譜的名稱、營養素名稱以及含量
    """
    i=0
    data_list=[0,0,0] #list用於存放搜尋資料，搜尋的比數需與list元素數量相符
    for r in R_id:
        try:
            sql="""
                SELECT recipename, nutrients, round(per100_cotent,2) FROM for_foodgroup.use_recipe_nutrients where r_id = %s and nutrients in ('Calories(kcal)', 'Carbohydrate(g)', 'Protein(g)', 'crude_fat(g)') order by nutrients;
            """ % r
            cursor.execute(sql)
            data = cursor.fetchall()
            data_list[i]= data
            i=i+1
            print(data_list)
        except:
            print("ERROR")




    """
    分別將食譜名稱放入list、營養素放入dictory
    """

    recipename = [data_list[0][0][0], data_list[1][0][0],data_list[2][0][0]]
    print(recipename)
    nutrients = {data_list[0][0][0]:[
                                    {"熱量(kcal)":data_list[0][0][2]},
                                    {"碳水化合物(g)":data_list[0][1][2]},
                                    {"脂質(g)":data_list[0][2][2]},
                                    {"蛋白質(g)":data_list[0][3][2]}
                                    ],
                 data_list[1][0][0]:[
                                    {"熱量(kcal)":data_list[1][0][2]},
                                    {"碳水化合物(g)":data_list[1][1][2]},
                                    {"脂質(g)":data_list[1][2][2]},
                                    {"蛋白質(g)":data_list[1][3][2]}
                                    ],
                 data_list[2][0][0]:[
                                    {"熱量(kcal)":data_list[2][0][2]},
                                    {"碳水化合物(g)":data_list[2][1][2]},
                                    {"脂質(g)":data_list[2][2][2]},
                                    {"蛋白質(g)":data_list[2][3][2]}
                                    ]
                }

    print(nutrients)

    """
    美化食譜名稱與營養素排版用於輪播系統呈現
    """

    nutrients_text_0 = "%s的每100克營養素含量\\n熱量(kcal): %s \\n碳水化合物(g): %s \\n脂質(g): %s \\n蛋白質(g): %s" %(recipename[0],nutrients[recipename[0]][0]["熱量(kcal)"],nutrients[recipename[0]][1]["碳水化合物(g)"],nutrients[recipename[0]][2]["脂質(g)"],nutrients[recipename[0]][3]["蛋白質(g)"])
    print(recipename[0])
    print(nutrients_text_0)
    print("-------------------")
    nutrients_text_1 = "%s的每100克營養素含量 \\n熱量(kcal): %s \\n碳水化合物(g): %s \\n脂質(g): %s \\n蛋白質(g): %s" %(recipename[1],nutrients[recipename[1]][0]["熱量(kcal)"],nutrients[recipename[1]][1]["碳水化合物(g)"],nutrients[recipename[1]][2]["脂質(g)"],nutrients[recipename[1]][3]["蛋白質(g)"])
    print(recipename[1])
    print(nutrients_text_1)
    print("-------------------")
    nutrients_text_2 = "%s的每100克營養素含量 \\n熱量(kcal): %s \\n碳水化合物(g): %s \\n脂質(g): %s \\n蛋白質(g): %s" %(recipename[2],nutrients[recipename[2]][0]["熱量(kcal)"],nutrients[recipename[2]][1]["碳水化合物(g)"],nutrients[recipename[2]][2]["脂質(g)"],nutrients[recipename[2]][3]["蛋白質(g)"])
    print(recipename[2])
    print(nutrients_text_2)


    """
    從"i_nutrition"資料庫中的"original_recipe"表格，根據R_id，取得食譜的名稱、圖片網址、網站連結網址
    """
    i=0
    data_list_2=[0,0,0]
    data_list=[0,0,0] #list用於存放搜尋資料，搜尋的比數需與list元素數量相符
    for r in R_id:
        try:
            sql="""
                SELECT RecipeName, RecipeURL, RecipeImageURL FROM i_nutrition.original_recipe where R_id = %s;
                """ % r
            cursor.execute(sql)
            data = cursor.fetchall()
            data_list_2[i]= data
            i=i+1
            print(data_list_2)
        except:
            print("ERROR")

    # db.close()



    """
    轉換食譜圖片連結、網站連結的資料格式
    """
    print(recipename)
    img_url = [data_list_2[0][0][2],data_list_2[1][0][2],data_list_2[2][0][2]]
    print(img_url)
    source_url =[data_list_2[0][0][1],data_list_2[1][0][1],data_list_2[2][0][1]]
    print(source_url)





    dataj = """
    [
    {
      "type": "template",
      "altText": "this is a carousel template",
      "template": {
        "type": "carousel",
        "actions": [],
        "columns": [
          {
            "thumbnailImageUrl": "%s",
            "text": "%s",
            "actions": [
              {
                "type": "uri",
                "label": "前往該食譜網頁",
                "uri": "%s"
              },
              {
                "type": "message",
                "label": "營養成分(每100克)",
                "text": "%s"
              }
            ]
          },
          {
            "thumbnailImageUrl": "%s",
            "text": "%s",
            "actions": [
              {
                "type": "uri",
                "label": "前往該食譜網頁",
                "uri": "%s"
              },
              {
                "type": "message",
                "label": "營養成分(每100克)",
                "text": "%s"
              }
            ]
          },
          {
            "thumbnailImageUrl": "%s",
            "text": "%s",
            "actions": [
              {
                "type": "uri",
                "label": "前往該食譜網頁",
                "uri": "%s"
              },
              {
                "type": "message",
                "label": "營養成分(每100克)",
                "text": "%s"
              }
            ]
          }
        ]
      }
    }
    ]""" % (img_url[0],
           recipename[0],
           source_url[0],
           nutrients_text_0,

           img_url[1],
           recipename[1],
           source_url[1],
           nutrients_text_1,

           img_url[2],
           recipename[2],
           source_url[2],
           nutrients_text_2
           )
    print(dataj)
    with open('./素材/顯示食譜推薦/reply.json', 'w', encoding='utf-8') as f:
        f.write(dataj)
        




    







