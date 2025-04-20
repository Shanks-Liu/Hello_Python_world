message = "Hello python world"
print(message)
message='heloo fucking world'
print(message)
print(message.upper())

first_name='ada'
last_name='lovelace'
full_name=first_name+' '+last_name
print('Hello, '+full_name.title()+' !')

name='python '
name=name.rstrip()
name

user_name="eric"
print('Hello '+user_name+':\n\twould you like to learn some Python today')

print('zhaoning said "you damn sucess to exam!"')
2**10                           
3/2

famous_person="zhao ning"
message="you damn sucess to exam!"
print(message)
 
print(3+5)
print(2**3)
print(4*2)
print(10-2)
 
import this
 
#学会了注释 
x=6
print('my favorate number x is:'+str(x))

print(12)
#学习列表
bicycle = ['gerk','price',12]
message='my favorate bicycle is a '+str(bicycle[2])+'.'
print(message)

names=['zhao ning','li chuan ming','wangjie']
friends = names[0].title()+' '+names[1].title()+' '+names[2].title()
print("Hello, friends: " + friends)

inventor=['tong yao','gao lu ping','song jing xin','aaasaida','boom']
print("Hi,would you like to have a dinner with me?\n\t"+str(inventor))
print(inventor[2].title()+': cant come for a cold,Im sorry for that')
inventor.pop(2)
inventor.insert(2,'liu xiao tong')
print("Hi,would you like to have a dinner with me?\n\t"+str(inventor))
del inventor[0]
print(inventor)
print(sorted(inventor,reverse=True)) #注意方法和函数的区别
#区分sort()方法（可加reverse） sorted()函数  reverse()方法
inventor.sort()
print(inventor)
inventor.reverse()
print(inventor)
len(inventor)
#学习操作列表
foods=['pissa','milk','seafood']
for food in foods:
    print(food+' I like this pepperoni pizza')
print('I really like pizza')
values=[]
for value in range(1,21):
    values.append(value)
values=[value*1 for value in range(1,1000001)] #列表解析
sum(values)
for value in range(1,20,2):
    print(value)
    
for value in range(1,5):
    print(value)    
valuess=value*2
values=[value**3 for value in range(1,11)]
test_names=names[:]
names.append('bai yu')
print('the first three items in the list are:')
for name in names[:3]:
    print('\t\t\t\t\t'+name)
test_names.append("liu tian jiao")
print("my favorate name is :")
for name in names:
    print(name)
print("my friends name is:")
for name in test_names:
    print(name)
print(names[0])
foods=('fish','chiken','hot dog','pig','rice')
for food in foods:
    print(food)
#核实错误
#foods[0]=shit


#学习if语句
car="bmw"
car=="bmw"
print("IS car =='bmw' ? I predict true")
print(car == 'bmw')
print("\n Is car == audi ? I predict true")
print(car == 'audi')

'sudi' == 'audi'
'sad'==2
car.upper()=="BMW"
'zhao ning'in names
alien_color='yellow'
if alien_color == 'green':
    print('you got 5 point')
elif alien_color == 'yellow':
    print('you got 10 point')
elif alien_color == 'red':
    print('you got 15 point')

current_users=['admin','lxl','zn','WJ','jfj']
if current_users:
    for current_user in current_users:
        if current_user == 'admin':
            print('Hello admin,would you like to see a status report?')
        else:
            print('hello ' + current_user + ' ,thank you for logging in again')
else:
    print('We need to find some users!')
new_users=['ZN','wj','xvz','xyj','JfJ']
test_names=[]
for current_user in current_users:
    test_names.append(current_user.lower())
for new_user in new_users:
    if new_user.lower() in test_names:
        print('you should change name')
    else:
        print('This name is available')
        
num_list=[1,2,3,4,5,6,7,8,9,'test']
for num in num_list:
    if num == 1:
        print('1st')
    if num == 2:
        print('2nd')
    if num > 2:
        print(str(num)+"th")
#恶意输入
age='test'
if age == 18:
    print("we won't let you in")
else:
    print("cool!")
    
#学习字典
zhao_ning = {
        'first_name': 'zhao',
        'last_name': 'ning',
        'age': 23,
        'city': 'tsingdao',
        }
wang_jie = {
        'first_name': 'wang',
        'last_name': 'jie',
        'age': 23,
        'city': 'qing dao',
        }
names = [zhao_ning,wang_jie]
for name in names:
    print("\n")
    for key,value in name.items():
        print(key + ": " + str(value))

#学习嵌套    
favorite_places = {
        'zhao ning': ['qing dao','wei fang','shang hai'],
        'wang jie': ['qing dao','huang dao','hong dao'],
        'liu yang': ['wei hai'],
        }
for name,places in favorite_places.items():
    if len(favorite_places[name]) != 1:
        print(name + "' favorite places are : " )
    else:
        print(name + "' favorite place is : ")
    for place in places:
        print('\t\t\t\t' + place)
        
python_dirs = {
        'title': 'show first name upper',
        'pop': 'delete the element in list',
        'print': ['print','output'],
        'sort': 'sort the element in list by initials order',
        'for': 'traverse the list',
        }
for key, value in python_dirs.items():
    print('\nmathod: ' + key)
    print('effect: ' + str(value))
print('all method is : ')
for method in sorted(python_dirs.keys()):
    print('\t'+ method)
values=[]
print('all effects is : ')
for value in python_dirs.values():
    if type(value) == str:
        values.append(value)
    else:
        for x in value:
            values.append(x)
sorted(values)
for y in values:
    print("\t"+y)
    
test_list = ['title','pop','for']
for method in python_dirs.keys():
    if method in test_list:
        print('i know this method')
    else:
        print("i don't know this method")
print(python_dirs['print'])


cities = {
        'qingdao': {
                'country': 'china',
                'population': '9000',
                'fact': 'bysea',
                },
        'tokyo':{
                'country': 'japan',
                'population': '8000',
                'fact': 'hot',
                },
        }
for city,city_infos in cities.items():
    print('introduce this city: ' + city)
#    print('country: ' + city_infos['country'])
#    print('population: ' + city_infos['population'])
#    print('fact: ' + city_infos['fact'])    
    for city_info1,city_info2 in city_infos.items():
        print('\t\t' + city_info1 + ': ' + city_info2)

#学习输入
car = input('What kind od car do you want rent: ')
print("Let me see if we can find you a " + car)
    
x = input('how many people in diet: ')
x = int(x)
if x >8:
    print('there is no empty desk')
else:
    print('there have avialable table')

active = True
while active:
    prompt = 'please enter what ingredients you want: '
    message = input(prompt)
    if message == 'quit':
        active = False
    else:
        print('we wil add this')
        
active = True
prompt = 'how old are you: '
x = input(prompt)
x = int(x)
if x < 3:
    price = 'free'
    print('your price is : ' + price)
if x <12:
    price = 10
    print('your price is : ' + str(price))
if x >= 12:
    price = 15
    print('your price is : ' + str(price))

x = 3
while x < 5 :
    print(x)

sandwich_orders = ['hot dog','humberge']
finished_sandwiches = []
while sandwich_orders:
    current_sandwich = sandwich_orders.pop()
    print('I made your tuna sandwich: ' + current_sandwich)
    finished_sandwiches.append(current_sandwich)
print('\n--- all sandwich ---\n')
for finished_sandwich in finished_sandwiches:
    print(finished_sandwich)
    
sandwich_orders = ['hot dog','pastrami','humberge','pastrami','pastrami']
print('our patrami are sold out')
while 'pastrami' in sandwich_orders:
    sandwich_orders.remove('pastrami')
print(sandwich_orders)

places = {}
active = True
while active:
    name = input("What's your name: ")
    place = input('If you could visit one place in the world, where would you go? ')
    places[name] = place
    respond = input('Would you like to let another person respond?(yes/no) ')
    if respond == 'no':
        active = False
print('\n--- poll result ---')
for name,place in places.items():
        print('\t' + name +  " want to go " + place)


#学习函数
def display_message():
    print('learn the function')
display_message()

def favorite_book(title):
    print('One of my favorite books is : '+ title.title())
favorite_book('Alice in Wonderland')

def make_shirt(size = 'XL',words = 'I love python'):
    print('The size is : ' + size + "\nThe words is : " + words)
make_shirt()
make_shirt('l')
make_shirt('l','I am your father')


def describe_city(name,country = 'China'):
    print(name + ' is in the ' + country)
describe_city('qing dao')
describe_city('shang hai')
describe_city('new york','America')


def get_formatted_name(first_name, last_name, middle_name=""):
    full_name = first_name + ' ' + middle_name + " " + last_name
    return full_name.title()
friend = get_formatted_name('wang','jie')
print(friend)


def make_album(singer_name, album, num=''):
    singer = {singer_name:album}
    if num:
        singer['num'] = num
    return singer
while True:
    print('(input quit to terminate this respond)')
    singer_name = input('The singer is : ')
    if singer_name == 'quit':
        break
    album = input('The album is : ')
    if album == 'quit':
        break
    num = input('The number of album is :')
    if num == 'quit':
        break
    for x,y in make_album(singer_name, album, num).items():
        print(x + ': ' + y)

   
def show_magicians(names):
    for name in names:
        print("Let's welcome " + name)
magicians_names =["jessica"]
great_name = []
def make_great(names):
    for i in range(len(names)):
        names[i] = 'the great ' + names[i] #修改列表里元素
        great_name.append(name)
make_great(magicians_names)
show_magicians(magicians_names)
show_magicians(great_name)

def make_pizza(*toppings):
    print("making a pizza with the fllowing toppings")
    for topping in toppings:
        print("-" + topping)
make_pizza("shit","cheese")

def build_profile(first,last,**user_info):
    '''describe the function'''
    profile = {}
    profile["first_name"] = first
    profile["last_nmae"] = last
    for key,value in user_info.items():
        profile[key] = value
    return profile
user_profile = build_profile("Shanks",
                             "Liu",
                             age = 23,
                             height =str(5.8) + "ft",
                             character= True,
                             )
print(user_profile)

import pizza
pizza.make_pizza(12, "cheese", "mushrooms" )


import sys  #python包路径
sys.path



from pizza import make_pizza as m_p
m_p(14, 'shit', 'cheese')

import pizza as p
p.make_pizza(13, 'cheese')

from pizza import *  #一般不使用,重名
make_pizza(18, 'cheese')

import print
unprint_names = ['robot', 'dragen']
complete_names = []
print.print_models(unprint_names, complete_names)
print.show_models(complete_names)
        

#学习类
class Restaurant():
    def __init__(self, restaurant_name, cuisine_type):
        self.restaurant_name = restaurant_name
        self.cuisine_type = cuisine_type
        self.number_served = 0
    def describe_restaurant(self):
        print(self.restaurant_name + " " + self.cuisine_type)
    def open_restaurant(self):
        print("The " + self.restaurant_name + " is opening")
    def read_number(self):
        print("The restaurant have " + 
              str(self.number_served) + 
              " people are eating.")
    def update_number(self,number):
        self.number_served += number
class IceCreamStand(Restaurant):
    def __init__(self,restaurant_name, cuisine_type, flavors):
        super().__init__(restaurant_name, cuisine_type)
        self.flavors = flavors
    def describe_icecream(self):
        for self.flavor in self.flavors:
            print(self.flavor)
flavors = ["orange", "strawberry"]
restaurant1 = IceCreamStand("wangjie_shaokao", "chinese_food", flavors)
restaurant1.describe_icecream()
print("The restaurant's name is " + restaurant1.restaurant_name +
      "\nand it's cuisine type is " + restaurant1.cuisine_type)
restaurant1.describe_restaurant()
restaurant1.open_restaurant()
restaurant1.update_number(2)
restaurant1.read_number()
restaurant2 = Restaurant("wang_pin_niu_pai", "Western food")
restaurant2.describe_restaurant()


class User():
    def __init__(self, first_name, last_name, *user_infos):
        self.first_name = first_name
        self.last_name = last_name
        self.login_attempts = 0
        self.full_name = first_name +" " + last_name
        self.user_infos = user_infos
    def describe_user(self):
        print("The user'name is: " + self.full_name)
        for self.user_info in self.user_infos:
            print("The user'information are: " + str(self.user_info))
    def greet_user(self):
        print("Hello! " + self.full_name)
    def increment_login_attempts(self):
        self.login_attempts += 1
    def reset_login_attempts(self):
        self.login_attempts = 0
    def read_login_attempts(self):
        print("Now, There have " + 
              str(self.login_attempts) + 
              " users is logging")
class Admin(User):
    def __init__(self, first_name, last_name, *user_infos):
        super().__init__(first_name, last_name, *user_infos)
        self.privileges = Privileges()
class Privileges():
    def __init__(self, privileges=privileges):
        self.privileges =privileges
    def show_privileges(self):
        for self.privilege in self.privileges:
            print(self.privilege)
privileges = ["can add post", "can delete post", "can ban users"]
user1 = Admin("Shanks", "Liu", 23, "male")
user1.privileges.show_privileges()
user1.describe_user()
user1.greet_user()
user1.read_login_attempts()
user1.increment_login_attempts()
user1.read_login_attempts()
user1.reset_login_attempts()
user1.read_login_attempts()
user2 =  User("LU", "Fei", 23, "Male", "niubi")
user2.describe_user()


class Car():
    def __init__(self, make, model, year):
        self.male = make
        self.model = model
        self.year = year
        self.odometer_reading = 0
class ElectricCar(Car):
    def __init__(self, make, model, year):
        super().__init__(make, model, year)
        self.battery = Battery() 
class Battery():
    def __init__(self):
        self.battery_size = 70
        #self.battery_size = battery_size
    def describe_battery(self):
        print("This car has a " + str(self.battery_size) + "-KWh battery")
    def upgrade_battery(self):
        if self.battery_size != 85:
            self.battery_size = 85
        return self.battery_size#可有可无
    def get_range(self):
        if self.battery_size == 70:
            range = 240
        if self.battery_size == 85:
            range = 270
        message = "This car can go " + str(range) + " miles on a full charge"
        print(message)
my_car = ElectricCar("Neptune", "model", 2018)
my_car.battery.get_range()
my_car.battery.upgrade_battery()
my_car.battery.get_range()


from restaurant import Restaurant
restaurant1 = Restaurant("zhao_zhao_xian", "chinese")
restaurant1.describe_restaurant()

from user import User, Admin, Privileges
admin1 = Admin("Wang", "jie", 24, "male")
admin1.privileges.show_privileges()#为什么列表不在载入的类里面也可以准确运行

from admin import Admin, Privileges
admin1 = Admin("Wang", "jie", 24, "male")
admin1.privileges.show_privileges()

from collections import OrderedDict
my_dic = OrderedDict()
my_dic["first"] = "Shanks"
my_dic["second"] = "Liu"
for key,value in my_dic.items():
    print(key + " : " + value)



from random import randint

class Die():
    
    def __init__(self, num, sides=6):
        self.sides = sides
        self.num = num
        
    def roll_die(self):
        for n in range(self.num):
            x = randint(1, self.sides)
            print(x)
            
roll1 = Die(10,20)
roll1.roll_die()


#学习文件和异常
with open(r"D:\Anaconda3\python_work\text\pi_digits.txt") as file_object:
    contents = file_object.read()#read读取全部内容
    print(contents)

import os  #当前工作目录  为什么会在C盘，能移动到python_work中吗
os.getcwd()

file_name = r"D:\Anaconda3\python_work\text\pi_digits.txt"
with open (file_name) as file:
    lines = file.readlines()
for line in lines:
    print(line.rstrip())


with open (r"python_practice\in_python.txt") as file:
    lines = file.readlines()
    content = file.read()

content.replace("python", "C")
print(content)

for line in lines:
    print(line.replace("python", "C").rstrip())

pi_string = ""
for line in lines:
    pi_string += line
    
print(pi_string.replace("python", "C").strip())

with open (r"python_practice\guest.text", "a") as file:
    active = True
    while active:
        x = input("Please enter your name: ")
        if x =='quit':
            break
        y = input("Why do you like programming: ")
        if y =="quit":
            break
        file.write(x + " log in \n")
        print("Hello, " + x)
        with open (r"python_practice\programming.text", "a") as file1:
            file1.write(y + "\n")


print("Please enter two number to addition: ")
print("Enter q to quit")
active = True
while active:
    while True:
        x = input("The first number: ")
        if x == "q":
            active = False
            break
        try:
            int_x = int(x)
        except ValueError:
            print("Please enter number not string")
            continue
        else:
            break
    while True:
        if x == "q":
            break
        y = input("The second number: ")
        if y == "q":
            active = False
            break
        try:
            int_y = int(y)
        except ValueError:
            print("Please enter number not string")
            continue
        else:
            break
    if x == 'q' or y == "q":
        break
    z = int_x + int_y
    print("\nThe answer is: " + str(z))


def print_animals(file):
    try:
        with open (file_name) as file:
            x = file.read()
    except FileNotFoundError:
        pass
    else:
        words = x.split()
        y = x.lower().count("the")
        sum = len(words)
        print("The file " + str(file)  + " have " + str(sum) + "words. " + 
              "And the word 'the' in this have show the " + str(y) + " times.")
        #用file_name而不是用file,file文件格式是 _io.TextIOWrapper
file_names = [r'python_practice\cats.txt', r"python_practice\sherlock.txt"]
for file_name in file_names:
    print_animals(file_name)



import json

nums = []
file_name = r"python_practice\file.json"#放在函数里会出现文件名未定义错误

def get_new_number():
    while True:

        num = input("Which number is your favorate_number: ")
        with open (file_name, "w") as file1:
            try:
                num_test = int(num)
            except ValueError:
                print("Please enter a number! ")
                continue
            else:
                nums.append(num)
                json.dump(nums, file1)
                print("We will remember your favorate number ")
                break
            
def get_stored_number():
    with open (file_name) as file1:
        nums = json.load(file1)

def enter_num():
    """让用户输入数字并指出"""
    while True:
        x = input("Does your favorate number have been log in ? (Y/N)  ")
        if x == "Y":
            try:
                get_stored_number()
            except FileNotFoundError:
                print("Now, there is no database, so we need your number ")
                get_new_number()
                break
            else:
                print("Welcome back, I konw your favorate number! " + 
                      "It's in this numbers: " + str(nums))
                break
        if x == "N":
            get_new_number()
            break
        else:
            print("Please enter Y or N")
            continue
enter_num()



print(5 , "hello")


























