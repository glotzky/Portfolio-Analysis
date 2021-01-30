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

    
