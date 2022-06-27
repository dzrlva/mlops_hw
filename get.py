#!/usr/bin/python

import psycopg2
from config import config

def get_by_id(product_id):
    conn = None
    row = None
    try:
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        cur.execute("SELECT * FROM items WHERE id = %s", (product_id,))
        print("The number of parts: ", cur.rowcount)
        row = cur.fetchone()
        '''
        while row is not None:
            print(row)
            row = cur.fetchone()
        '''
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
    return row

def get_by_parent_id(product_id):
    conn = None
    try:
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        cur.execute("SELECT * FROM items WHERE parentId = %s ORDER BY id", (product_id,))
        print("The number of parts: ", cur.rowcount)
        row = 0
        row = cur.fetchall()
        '''
        while row is not None:
            print(row)
            row = cur.fetchone()
        '''
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
    return row


