class user_build:
    def __init__(self, name, principal):
        "initialize the user"
        self.name = name #name or user
        self.principal = principal #principal amount invested
        self.total = 0 #total total amount of money invested
        self.percent = 0 # percent
    
    def display_userdata(self):
        """ Returns user data"""
        userdata = {'name':self.name, 'principal investment': self.principal, 'total investment': self.total,'Total Percentage gained': self.percent
               }
        return userdata

    

class Person:   
      
    # init method or constructor    
    def __init__(self, name):   
        self.name = name   
      
    # Sample Method    
    def say_hi(self):   
        print('Hello, my name is', self.name)   
  
# Creating different objects      

p1 = Person('Nikhil')   
p2 = Person('Abhinav') 
p3 = Person('Anshul') 
  
p1.say_hi()   
p2.say_hi() 
p3.say_hi()