package main

import (
  "fmt"
  //"structpb"
//  "cloud.google.com/go"
  "context"
  aiplatform "cloud.google.com/go/aiplatform/apiv1beta1"
  aiplatformpb "google.golang.org/genproto/googleapis/cloud/aiplatform/v1beta1"
  "google.golang.org/protobuf/types/known/structpb"
)

type PredictRequest struct {

	Endpoint string `protobuf:"bytes,1,opt,name=endpoint,proto3" json:"endpoint,omitempty"`
	Instances []*structpb.Value `protobuf:"bytes,2,rep,name=instances,proto3" json:"instances,omitempty"`
	Parameters *structpb.Value `protobuf:"bytes,3,opt,name=parameters,proto3" json:"parameters,omitempty"`
}


func main() {
  fmt.Printf("In vertex prediction")
  ctx := context.Background()
  c, err := aiplatform.NewPredictionClient(ctx) 
  if err != nil {
		// TODO: Handle error.
	}
  defer c.Close()
  endpoint := "https://us-central1-aiplatform.googleapis.com/v1/projects/demogct/locations/us-central1/endpoints/3491529960328265728:predict"
  
//  instances, err := structpb.NewValue(map[string]interface{}{"instances": []interface{}{[]float64{0.14817968167448203, 1.627787546998503, 0.32235801932243646, -0.9273702009469922, 2.2508163678305717, 1.4992889740999278, 0.3799797947268641, 0.17466176131179917, 1.6040410238866307, 0.32761173744624417, 0.0, 0.0, 0.0, -0.2041241452319315, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.20412414523193148, -0.29488391230979427, -0.20412414523193148, 0.0, 0.0, -0.14285714285714288, 0.0, 0.0, -0.14285714285714288, 0.0, 0.0, -0.14285714285714288, -0.43643578047198484, -0.25264557631995566, 0.0, 0.0, 0.0, 0.0, 1.8829377433825434, -0.20412414523193148, 0.0, -0.25264557631995566, -0.29488391230979427, 0.0, 0.0, 0.0, 0.0, -0.14285714285714288, -0.20412414523193148, -0.25264557631995566, -0.14285714285714288, -0.8164965809277261, -0.3692744729379982, 1.0834726777719228}}})

instances_val, err := structpb.NewValue(map[string]interface{}{"instances": []interface{}{[]interface{}{0.14817968167448203, 1.627787546998503, 0.32235801932243646, -0.9273702009469922, 2.2508163678305717, 1.4992889740999278, 0.3799797947268641, 0.17466176131179917, 1.6040410238866307, 0.32761173744624417, 0.0, 0.0, 0.0, -0.2041241452319315, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.20412414523193148, -0.29488391230979427, -0.20412414523193148, 0.0, 0.0, -0.14285714285714288, 0.0, 0.0, -0.14285714285714288, 0.0, 0.0, -0.14285714285714288, -0.43643578047198484, -0.25264557631995566, 0.0, 0.0, 0.0, 0.0, 1.8829377433825434, -0.20412414523193148, 0.0, -0.25264557631995566, -0.29488391230979427, 0.0, 0.0, 0.0, 0.0, -0.14285714285714288, -0.20412414523193148, -0.25264557631995566, -0.14285714285714288, -0.8164965809277261, -0.3692744729379982, 1.0834726777719228 } }})

  req := &aiplatformpb.PredictRequest{
          {Endpoint:endpoint, Instances:instances_val, Parameters:nil},      
	}
  resp, err := c.Predict(ctx, req)
  if err != nil {
		// TODO: Handle error.
	}
	// TODO: Use resp.
	_ = resp  
}
