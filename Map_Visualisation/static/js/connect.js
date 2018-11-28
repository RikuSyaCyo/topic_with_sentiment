function post(URL, PARAMS,f) {
    var result;
    console.log("in post");
    console.log("post:"+PARAMS);
    $.ajax({
        type: "POST",
        url: URL,
        data: {name:PARAMS},
        cache: false,
       // async: false,
        //contentType: "application/json", 
        //dataType: "json",
        success: function (data) {
            f(data);
            //result = data;
        }
    });
    console.log("leave post");
   // return result;
}