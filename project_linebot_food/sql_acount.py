from sqlalchemy import create_engine
import pymysql

def acount2():  # 套件 pymysql 時使用, db = "for_foodgroup"
    host = ""
    port = 3306
    user = ""
    password = ""
    db = ""
    connection = pymysql.connect(host=host, port=port, user=user, passwd=password, db=db, charset="utf8")
    cursor = connection.cursor()
    return cursor

