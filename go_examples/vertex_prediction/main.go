package main

import (
//  "fmt"
  "encoding/json"
  "fmt"
  "github.com/jmcvetta/napping"
  "log"
  "net/http"
)

func main() {

  url := "https://us-central1-aiplatform.googleapis.com/v1/projects/demogct/locations/us-central1/endpoints/3491529960328265728:predict"
  //fmt.Println("URL:>", url)
  s := napping.Session{}
    h := &http.Header{}
    h.Set("Content-Type", "application/json")
    h.Set("Authorization", "Bearer ya29.A0ARrdaM-0FJj2AD4JgWqrBZW7-YRSVhbq1tF3pL3purHdI2yDEo5fAZmFNIp6ouqIMB1qnLuTWPnmrfHU-Jww3MzhhtV9GUUX5uIpAebV0HWtsCq2NpZ4RyZxUNH4vZTi_FmPG6RJ0zvKhU14muTAGHBe0p-VtR8GBfo69A")
    s.Header = h
     var jsonStr = []byte(`
   {
    "instances": [[0.14817968167448203, 1.627787546998503, 0.32235801932243646, -0.9273702009469922, 2.2508163678305717, 1.4992889740999278, 0.3799797947268641, 0.17466176131179917, 1.6040410238866307, 0.32761173744624417, 0.0, 0.0, 0.0, -0.2041241452319315, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.20412414523193148, -0.29488391230979427, -0.20412414523193148, 0.0, 0.0, -0.14285714285714288, 0.0, 0.0, -0.14285714285714288, 0.0, 0.0, -0.14285714285714288, -0.43643578047198484, -0.25264557631995566, 0.0, 0.0, 0.0, 0.0, 1.8829377433825434, -0.20412414523193148, 0.0, -0.25264557631995566, -0.29488391230979427, 0.0, 0.0, 0.0, 0.0, -0.14285714285714288, -0.20412414523193148, -0.25264557631995566, -0.14285714285714288, -0.8164965809277261, -0.3692744729379982, 1.0834726777719228]]
}`)


    var data map[string]json.RawMessage
    err := json.Unmarshal(jsonStr, &data)
    if err != nil {
        fmt.Println(err)
    }
    fmt.Printf(string(json.Marshal(&data)))
    resp, err := s.Post(url, &data, nil, nil)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println("response Status:", resp.Status())
    fmt.Println("response Headers:", resp.HttpResponse().Header)
    fmt.Println("response Body:", resp.RawText())
}



