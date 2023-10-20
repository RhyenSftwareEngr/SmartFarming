//
//  demo1.swift
//  Project
//
//  Created by slu ictr ilab 4 on 9/14/23.
//

import Foundation

public class UserInfo {
        let firstName = " "
        let lastName = " "
        let username = " "
        let email = " "
        let password = " "
        let phoneNum: Int
        let userProfile: Bool

        
/**
 Reads a String inputs from the user to get the basic information of a user
 */
    func getUserInfo(){
        print("First Name: ", terminator: "")
        firstName = readLine()!
        print("Last Name: ", terminator: "")
        lastName = readLine()!
        print("Username: ", terminator: "")
        username = readLine()!
        print("Phone Number: ", terminator: "")
        phoneNum = Int (readLine()!)!
        print("Email: ", terminator: "")
        email = readLine()!
        print("password: ", terminator: "")
        password = readLine()!
    }
    
    func userRole(){
        print("1. As a Buyer")
        print("2. As a Seller")
        print("Enter choice: ", terminator: "")
    }
}
