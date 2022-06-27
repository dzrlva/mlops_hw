#!/usr/bin/python

import psycopg2
from config import config


def insert_vendor(type, name, id, price, parentId, date):
    """ insert a new vendor into the vendors table """
    sql_type = """INSERT INTO ITEMS(type, name, id, price, parentId, date)
             VALUES(%s, %s, %s, %s, %s, %s) RETURNING type;"""
    sql_name = """INSERT INTO ITEMS(name)
             VALUES(%s) RETURNING name;"""
    sql_id = """INSERT INTO ITEMS(id)
             VALUES(%s) RETURNING id;"""
    sql_price = """INSERT INTO ITEMS(price)
             VALUES(%s) RETURNING price;"""
    sql_parentId = """INSERT INTO ITEMS(parentId)
             VALUES(%s) RETURNING parentId;"""
    sql_date = """INSERT INTO ITEMS(date)
             VALUES(%s) RETURNING date;"""

    conn = None
    vendor_id = None
    try:
        # read database configuration
        params = config()
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**params)
        # create a new cursor
        cur = conn.cursor()
        # execute the INSERT statement
        cur.execute(sql_type, (type, name, id, price, parentId, date ))
#        cur.execute(sql_name, (name,))
#        cur.execute(sql_id, (id,))
#        cur.execute(sql_price, (price,))
#        cur.execute(sql_parentId, (parentId,))
#        cur.execute(sql_date, (date ))
        # get the generated id back
        vendor_id = cur.fetchone()[0]
        # commit the changes to the database
        conn.commit()
        # close communication with the database
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
    return vendor_id


def update(type, name, id, price, parentId, date):
    """ insert multiple vendors into the vendors table  """
    sql = """UPDATE ITEMS SET type=%s, name=%s, price=%s, parentId=%s, date=%s WHERE id = %s"""
    conn = None
    try:
        # read database configuration
        params = config()
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**params)
        # create a new cursor
        cur = conn.cursor()
        # execute the INSERT statement
        cur.execute(sql, (type, name, price, parentId, date, id))
        # commit the changes to the database
        conn.commit()
        # close communication with the database
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

if __name__ == '__main__':
    # insert one vendor
    print(insert_vendor("3M Co."))
    insert_vendor_list([
        ('AKM Semiconductor Inc.',),
        ('Asahi Glass Co Ltd.',),
        ('Daikin Industries Ltd.',),
        ('Dynacast International Inc.',),
        ('Foster Electric Co. Ltd.',),
        ('Murata Manufacturing Co. Ltd.',)
    ])
