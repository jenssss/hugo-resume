+++
title = "Automatically update calendars from Gmail"
author = ["Jens Svensmark"]
date = 2020-04-04
tags = ["howto", "google-apps-script", "Gmail", "calendar", "TimeTree", "regexp"]
draft = false
math = true
#comment = "this file is [auto-generated]"
+++

How to automatically create events in Google and TimeTree calendars
using regular expression matching with Google Apps Script.

<!--more-->


## The problem {#the-problem}

I like to cook, but I'm too lazy to plan what to make, and do the
grocery shopping for it. When I was living in the US, I found a meal
delivery service (Blue Apron), where I could pick meals on their
homepage, and then they would deliver a box with all the needed
ingredients and a recipe. A perfect solution for me, since I could get
to do the cooking, but didn't have to worry about thinking up recipes,
or planning the grocery shopping.

After I moved to Japan I found a similar service called
[Yoshikei](http://yoshikei-dvlp.co.jp/). Different days have different recipes, so every week I pick
two or three days with recipes I think look good. The problem then
becomes that I can never remember for which days I ordered the meal
boxes, and going to the Yoshikei homepage to check my order history is
a bit of a hassle. Ideally I want to be able to check which days I get
the deliveries in my calendar, but I'm too lazy to write this in every
week. I felt like there ought to be a way to automate this.

I noticed that in the email order confirmations from Yoshikei there
would be a list of dishes and delivery dates, and this list would
always be in the same format, like in this excerpt

```text
下記内容が、ご注文週の最終注文内容となります。
-----------------------------------------------------------------------------
2021/04/06（火）カットミール２人用（217）　ハニーマスタードチキン：1セット
2021/04/07（水）カットミール２人用（317）　簡単！焼きカレー：1セット
2021/04/08（木）カットミール２人用（417）　揚げない♪えびマヨ：1セット
```

So it shouldn't be difficult to extract the date and the name of the
dish[^fn:1], to insert this into a calendar. But how to connect this to
the email in an automated way?

Enter [Google Apps Script](https://www.google.com/script/start/). Apps Script is an online platform, where you
can write scripts in JavaScript. Since it's a Google service, it comes
with integrations for handling Gmail and Google Calendar, as well as a
number of other Google services. It also has triggers for
automatically executing scripts, for instance once per week.

This was my first foray into writing anything with JavaScript, so it
took a fair bit of googling, but I was eventually able to knit
together a working solution.


## The solution {#the-solution}


### Reading the email {#reading-the-email}

First I setup a filter in Gmail to add the label `Receipts/Yoshikei`
to all emails coming from Yoshikei. Next I wrote the following
function in Apps Script

```javascript
function readFoods(){
    var yoshiLabel = GmailApp.getUserLabelByName("Receipts/Yoshikei");
    var yoshiThreads = yoshiLabel.getThreads(0, 1);
    var yoshiThread = yoshiThreads[0];
    var foods = getFoodsFromThread(yoshiThread);
    return foods;
}
```

A couple of things going on here, so let's unpack them one by one.
First I use the `GmailApp` object. This object provides an interface
to Gmail, and is documented [here](https://developers.google.com/apps-script/reference/gmail/gmail-app). From this I get all messages with
the `Receipts/Yoshikei` label. In Gmail, messages are organized into
threads, so I retrieve the first thread in this label, and sends it
to another function for processing

```javascript
function getFoodsFromThread(yoshiThread){
    var messages = yoshiThread.getMessages();
    var message = messages[0];
    var foods = getFoods(message);
    return foods;
}
```

In this function I simply get the messages from the thread, and send
the first message on for further processing

```javascript
function getFoods(message) {
    var matches = [];
    var divideMatchString = "下記内容が、ご注文週の最終注文内容となります";
    var n = message.search(divideMatchString);
    if (n != -1){
	message = message.substring(n);
	// This regexp matches to a date in yyyy/mm/dd format, then parenthesis with any character followed by line break
	// After that any character until line break are matched
	var matchString = "([0-9]{4}/[0-9]{2}/[0-9]{2}（.）)(.*)";
	var match = message.match(matchString);
	var i=0;
	while(match != null && i<10){
	    var dateString = match[1];
	    var foodString = match[2];
	    matches.push([dateString, foodString]);
	    message = message.substr(match.index+match[0].length);
	    match = message.match(matchString);
	    i = i + 1;
	}
    }
    return matches;
}
```

This function uses regular expression matching to extract the
information I want from the message. First I look for the string `下
記内容が、ご注文週の最終注文内容となります` using the javascript
built in string `search` method. This string comes immediately before
the section of the email I am interested in, so I remove any part
of the message before this part using the `substring` method. Next I
use a regular expression to match to a line that starts with a
date. If a match is found, it is added to the an array called
`matches`. This search is repeated until no more matches are found (or
until it has been run 10 times).

Next, I wrote functions for extracting the date as a `Date` object,
and the part of the string that contains the name of the food.

```javascript
function parseFood(food){
    const [date, endTime] = parseFoodDate(food[0]);
    const foodName = parseFoodName(food[1]);
    return [date, endTime, foodName];
}

function parseFoodDate(dateString){
    var miliSecsPerHour = 3600*1000;
    var date = new Date(Date.parse(dateString.substr(0, 10))+19*miliSecsPerHour);
    var endTime = new Date(date.getTime()+2*miliSecsPerHour);
    return [date, endTime];
}

function parseFoodName(foodString){
    var foodArray = foodString.replace(/\s+/g, " ").replace("：", " ").split(" ");
    var foodName = foodArray[foodArray.length - 2];
    foodName = foodName.split("（")[0]
    return foodName;
}
```


### Syncing to the calendars {#syncing-to-the-calendars}

Now, to put this food into Google Calendar. Since I was using Google
Apps Script, this was fairly easy to do using the [CalendarApp](https://developers.google.com/apps-script/reference/calendar/calendar-app) object.
I got the calendar ID from the settings page in Google Calendar (here
I replaced the ID with asterisks). I am also checking that I didn't
already create an event for Yoshikei on the relevant day, before
adding the new event.

```javascript
function putFoodInGCalendar(food){
    const [date, endTime, foodName] = parseFood(food);

    var calId = "**********";
    var cal = CalendarApp.getCalendarById(calId);
    var eventThatDay = cal.getEventsForDay(date, {search:"ヨシケイ"});
    if (eventThatDay.length == 0) {
	cal.createEvent(foodName, date, endTime, {"description": "ヨシケイ"});
	Logger.log(date);
	Logger.log("Created event");
    } else {
	Logger.log(date);
	Logger.log("Event already exists");
    }
}
```

I also wanted to add an event into a different calendar app called
[TimeTree](https://timetreeapp.com/). Fortunately this app has a [public API](https://developers.timetreeapp.com/en/docs/api/overview), but since it is not a
Google service, interacting with the API takes a bit more work than
for Google Calendar. It can be done though, using the [UrlFetchApp](https://developers.google.com/apps-script/reference/url-fetch/url-fetch-app)
class, as in the following

```javascript
function putFoodInTimetree(food){

    const [date, endTime, foodName] = parseFood(food);

    var AccessToken = "**********";
    var CalendarID = "**********"
    var me_ID1 = "**********";
    var me_ID2 = "**********";
    var me = {"id":me_ID1,"type":"user"};
    var headers = {"Accept" : "application/vnd.timetree.v1+json",
		   "Authorization" : "Bearer "+AccessToken};

    var event_data = {
	"data": {
	    "attributes": {
		"category": "schedule",
		"title": foodName,
		"all_day": false,
		"start_at": date.toISOString(),
		"end_at": endTime.toISOString(),
		"description": "ヨシケイ",
		"location": "Home",
		"url": "https://www2.yoshikei-dvlp.co.jp/webodr/"
	    },
	    "relationships": {
		"label": {
		    "data": {
			"id": meID2,
			"type": "label"
		    }
		},
		"attendees": {
		    "data": [
			me
		    ]
		}
	    },
	}
    };

    var event_data_str = JSON.stringify(event_data);
    headers["Content-Type"] = "application/json;"
    var options = {"method": "POST",
		   "headers": headers,
		   "payload": event_data_str,
		   "muteHttpExceptions": true};
    var response = UrlFetchApp.fetch("https://timetreeapis.com/calendars/"+CalendarID+"/events", options);
    var insertedEvent = response.getContentText();
    Logger.log(insertedEvent);
}
```

Finally, to pull all this together, I wrote the main entry point function
for my project

```javascript
function readFoodAndPutInCalendar() {
    var foods = readFoods();
    for (var i = 0; i < foods.length; i++) {
	if (isThisANewFood(foods[i])){
	    putFoodInGCalendar(foods[i]);
	    putFoodInTimetree(foods[i]);
	}
    }
}
```

When setting up a trigger for automatically running this project, this
is the function that the trigger should run. The function can also be
run manually in the Apps Script interface.

As a security feature, Google Apps Script requires that [authorization
scopes](https://developers.google.com/identity/protocols/oauth2/scopes) for the script be specified. This should happen automatically,
but in case it doesn't, one can manually specify these scopes in the
`appsscript.json` file. For this project the required scopes are

```json
{
  "oauthScopes": [
      "https://www.googleapis.com/auth/gmail.readonly",
      "https://www.googleapis.com/auth/calendar.events",
      "https://www.googleapis.com/auth/calendar.readonly",
      "https://www.googleapis.com/auth/script.external_request"
    ],
}
```

I made a [gist](https://gist.github.com/jenssss/1d17319085f89c91f5967c518b08fac0) with all the code from this post, and an additional
feature designed to prevent accidentally adding the same dish more
than once.

[^fn:1]: For those not so proficient in Japanese, the name of the dish is the text that appears after the parenthesis with a number inside, until the colon. So the first dish is "ハニーマスタードチキン", which is "honey mustard chicken" in English.
