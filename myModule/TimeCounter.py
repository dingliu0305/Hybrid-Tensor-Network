import time
import traceback

# 异常捕获装饰器
# error catch decorator
def catch_except(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            func()
            span = time.time() - start
            print('successfully execute!   time span: ',sec2time(span))
        except Exception as e:
            span = time.time() - start
            print("error message: ",e,"   time span: ",sec2time(span))
    return wrapper

 # 将获得的整型时间格式化
def format_time2str(hour,minute,second):
    if(not isinstance(hour, int) or not isinstance(minute, int) or not isinstance(second, int)):
        raise ValueError("wrong input type")
    if 0 <= second < 10:
        second = '0' + str(second)
    else:   
        second = str(second)
    if 0 <= minute < 10:
        minute = '0' + str(minute)
    else:
        minute = str(minute)
    if 0 <= hour < 10:
        hour = '0' + str(hour)
    else:
        hour = str(hour)
    return ':'.join([hour,minute,second])

# 将秒转化为完整的时间（年月日时分秒）
def sec2time(second):
    new_second = int(second % 60)
    minute = int(second / 60) % 60
    hour = int(second / 60 / 60) % 60
    return format_time2str(hour, minute, new_second)

@catch_except
def test():
    print('test start !')
    li = [1,2,0,32,1,231,3,123,123,12312,0]
    for item in li:
        a = 2 / item
        time.sleep(2)
    print('test end !')

if __name__ == "__main__":
    test()
    
    
    
