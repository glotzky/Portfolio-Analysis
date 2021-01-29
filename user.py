from build_portfolio import *
def start():
    name = input('Name: ')
    principal = int(eval(input('Principal amount invested: ')))
    menu()

def menu():
    option = int(eval(input('Select an option: ')))

def get_info():
    user = user_build(name,principal)
    return user.display_userdata()

start()
