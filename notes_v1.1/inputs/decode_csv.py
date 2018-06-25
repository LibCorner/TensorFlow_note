# -*- coding: utf-8 -*-
import tensorflow as tf


'''
tf.decode_csv(
    records,
    record_defaults,
    field_delim=',',
    use_quote_delim=True,
    name=None,
    na_value=''
)

records: A Tensor of type string. Each string is a record/row in the csv and all records should have the same format.
record_defaults: A list of Tensor objects with specific types. Acceptable types are float32, float64, int32, int64, string. One tensor per column of the input record, with either a scalar default value for that column or empty if the column is required.
field_delim: An optional string. Defaults to ",". char delimiter to separate fields in a record.
use_quote_delim: An optional bool. Defaults to True. If false, treats double quotation marks as regular characters inside of the string fields (ignoring RFC 4180, Section 2, Bullet 5).
name: A name for the operation (optional).
na_value: Additional string to recognize as NA/NaN.
'''

s=["1,2,3,4,5"]*50



sess=tf.Session()

t=tf.convert_to_tensor(s)
print(sess.run(t))
res=tf.decode_csv(t,record_defaults=[[1.]]*5,field_delim=',')
print(sess.run(res))
