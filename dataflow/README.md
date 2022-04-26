## Dataflow Tutorial

Create a Maven Archtetype for Beam
     mvn archetype:generate \
     -DarchetypeGroupId=org.apache.beam \
     -DarchetypeArtifactId=beam-sdks-java-maven-archetypes-examples \
     -DarchetypeVersion=2.22.0 \
     -DgroupId=org.example \
     -DartifactId=first-dataflow \
     -Dversion="0.1" \
     -Dpackage=org.apache.beam.examples \
     -DinteractiveMode=false
