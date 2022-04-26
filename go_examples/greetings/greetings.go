package greetings

import ("errors"
         "fmt" 
         "math/rand"
         "time"
)

func Hello(name string) (string, error) {

 if name == "" {
   return "", errors.New("empty name")
  }
 message := fmt.Sprintf(randomFormat(), name)
 return message, nil 

}

func int(){
//  rand.Seed(time.Now().UnixNano())
}

func randomFormat() string{
  //A slice of message format
  formats := []string{
     "Hi %v Welcome",
     "Great to see you, %v",
     "Namaste, %v, what a beautiful dress you have",
     "Yankee %v","Namm %v what man",
    }
  fmt.Println(len(formats))
  rand.Seed(time.Now().UnixNano())
  fmt.Println(rand.Intn(len(formats)))
  return formats[rand.Intn(len(formats))]

}


