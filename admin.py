# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 19:47:35 2018

@author: 游侠-Speed
"""
from user import User

class Admin(User):
    def __init__(self, first_name, last_name, *user_infos):
        super().__init__(first_name, last_name, *user_infos)
        self.privileges = Privileges()
        
        
class Privileges():
    def __init__(self, privileges=privileges):
        #直接在初始化函数里赋值会出错，要想不出错，要先执行最后一行，变量里面有了列表后，再执行前面代码就可以，或者在初始化函数里不包括该属性，在下面再定义默认值
        privileges = ["can add post", "can delete post", "can ban users"]
        self.privileges = privileges
        
    def show_privileges(self):
        for self.privilege in self.privileges:
            print(self.privilege)
