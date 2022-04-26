package main

import (
	"fmt"
        "log"
	"example.com/greetings"
)

func main(){
    
   log.SetPrefix("greetings:")
   log.SetFlags(0)


   //Request a empty greeting message that triggers an error
    message, err := greetings.Hello("Saraswati")
   //message := greetings.Hello("Sheetal")
    if err != nil { 
       log.Fatal (err)
    }
    fmt.Println(message)

}
