# twitter_profiler
Module for profiling users on twitter. Written in Python.

At the moment it only determines the gender of a given user. 
The expected input is a json file with a list of Twitter's API user objects. 

It tries to recognize the user's names and from that determine its gender. 
If no name could be found it fetches the profile's image and classifies the image as a photo-like or logo-like. The hypothesis being that Non-People accounts will have a logo. Contrary to People that will have a picture. 
