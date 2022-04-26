package main

import (
	aiplatform "cloud.google.com/go/aiplatform/apiv1beta1"
	"context"
	"fmt"
        "encoding/json"
	"google.golang.org/api/option"
	//"google.golang.org/api/option/internaloption"
	aiplatformpb "google.golang.org/genproto/googleapis/cloud/aiplatform/v1beta1"
	structpb "google.golang.org/protobuf/types/known/structpb"
)

func main() {
	ctx := context.Background()
	c, err := aiplatform.NewPredictionClient(ctx,option.WithEndpoint("us-central1-aiplatform.googleapis.com:443") )
	if err != nil {
		fmt.Println("Error: ", err)
	}
	defer c.Close()

	instances, _ := structpb.NewValue(
		map[string]interface{}{
			"instances": []interface{}{
				[]interface{}{0.14817968167448203, 1.627787546998503, 0.32235801932243646, -0.9273702009469922, 2.2508163678305717, 1.4992889740999278, 0.3799797947268641, 0.17466176131179917, 1.6040410238866307, 0.32761173744624417, 0.0, 0.0, 0.0, -0.2041241452319315, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.20412414523193148, -0.29488391230979427, -0.20412414523193148, 0.0, 0.0, -0.14285714285714288, 0.0, 0.0, -0.14285714285714288, 0.0, 0.0, -0.14285714285714288, -0.43643578047198484, -0.25264557631995566, 0.0, 0.0, 0.0, 0.0, 1.8829377433825434, -0.20412414523193148, 0.0, -0.25264557631995566, -0.29488391230979427, 0.0, 0.0, 0.0, 0.0, -0.14285714285714288, -0.20412414523193148, -0.25264557631995566, -0.14285714285714288, -0.8164965809277261, -0.3692744729379982, 1.0834726777719228}}})
        obj, err := json.Marshal([]*structpb.Value{instances})
        fmt.Printf(string(obj))
	req := &aiplatformpb.PredictRequest{
		Endpoint:   "projects/demogct/locations/us-centra1/endpoints/3491529960328265728",
		Instances:  []*structpb.Value{instances},
		Parameters: nil,
	}
	resp, err := c.Predict(ctx, req)
	if err != nil {
		fmt.Println("Error: ", err)
	}
	// TODO: Use resp.
	_ = resp
}
