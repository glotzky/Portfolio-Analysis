from build_portfolio import *
class User:
        


    def menu():
        print("Main Menu: ")
        print("1: Get User Info")
        print("2: See total return")
        print("3: Exit")
        option = int(eval(input('Select an option: ')))
        if option == 1:
            user.display_userdata()
        elif option == 2:
            print('hi')
        else:
            print('hi')
def start():
    name = input('Name: ')
    principal = int(eval(input('Principal amount invested: ')))
    user = user_build(name, principal)
    menu()

start()
name = 'patrick'
principal = 4000
user = user_build(name, principal)

