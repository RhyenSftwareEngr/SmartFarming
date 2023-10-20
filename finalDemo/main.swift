//
//  main.swift
//  Project
//
//  Created by slu ictr ilab 4 on 9/14/23.
//


import Foundation


print("APPDEV PROJECT")

loginMenu()

print("Enter choice: ", terminator: "")
var choice: Int = getInt()
switch choice {
case 1:
    
    let userObject = UserInfo()
    userObject.userRole()
    choice = getInt()
    userObject.getUserInfo()
  
    repeat{
        if choice == 1 {
            let buyerObject = BuyerPage()
            buyerObject.mainMenu()
        } else if choice == 2 {
            let sellerObject = SellerPage()
            sellerObject.mainMenu()
        } else {
            print("Invalid input! Please choose from the given choices.")
        }
    }while choice != 1 && choice != 2

case 2:
    print("Case 2")
default:
    print("Please enter a number")
}

/**
 Displays the lgin menu choices
 */
func loginMenu() {
    print("1. Register")
    print("2. Log-in")
}



/**
 Reads an input from the user then unwrapped and convert into Integer to be returned
 */
func getInt() -> Int {
    let input = readLine()
    return Int (input!)! //unwrapping the input then converts into integer
}
