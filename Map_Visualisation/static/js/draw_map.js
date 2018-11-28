var p_map = {};
var t_map = {};
var minvalue = 0;
var maxvalue = 0;

function initial_map(){
var width = 1000;
var height = 1000;
var left = 30;
var right = 30;
var projection = d3.geo.mercator()
                      .center([-100, 30])
                      .scale(850)
                      .translate([width/2, height/2]);
 var svg = d3.select("body").append("svg")
    .attr("class","map")
      .attr("width", width)
      .attr("height", height)
      .style("position","absolute")
      .style("left",left)
      .style("top",top)
      .append("g")
      .classed("content",true)
      .append("g");
  
  
  var color = d3.scale.category20();
  var path = d3.geo.path()
    .projection(projection);

  var tooltip = d3.select("body")
    .append("div")
    .attr("class", "tooltip")
    .style("opacity", 0.0);
  
  d3.json("static/usa.json", function(error, root) {
        
    if (error) 
        return console.error(error);
    //console.log(root.features);
        
    svg.selectAll("path")
        .data( root.features )
        .enter()
        .append("path")
        .attr("stroke","#ffffff")
        .attr("stroke-width",0.5)
        .attr("fill", "#EBEBEB")
        .attr("d", path )   //使用地理路径生成器
        .attr("class", function(d){
          return d.properties.postal;
        })
        // .on("mouseover",function(d,i){
        //     d3.select(this)
        //        .attr("fill","#D3D3D3");
        // })
        // .on("mouseout",function(d,i){
        //     d3.select(this)
        //        .attr("fill", "#EBEBEB");
        // });

    svg.selectAll("text")
      .data(root.features)
      .enter()
      .append("text")
      .attr("class", function(d){
      	return d.properties.postal + "_text"
      })
      .html(function(d){
        return d.properties.postal;
      })
      .attr("x", function(d){
        var position = [d.properties.longitude, d.properties.latitude]
        var pro_position = projection(position)
        return pro_position[0]
      })
      .attr("y", function(d){
        var position = [d.properties.longitude, d.properties.latitude]
        var pro_position = projection(position)
        return pro_position[1]
      })
      .attr("fill", "#000000")
      .attr("font-size", "8");

	var linear = d3.scale.linear()
	              .domain([minvalue, maxvalue])
	              .range([0,1]);
	var maxcolor = d3.rgb(255, 69, 0);
	var mincolor = d3.rgb(255, 255, 255);
	var computeColor = d3.interpolate(mincolor, maxcolor);
	for(var key in p_map){
	  var t = linear(p_map[key]);
	  var color = computeColor(t);
	  d3.select("."+key)
	    .attr("fill", color.toString())
	    .on("mouseover", function(d,i){
	      key_now = d3.select(this).attr("class");
	      tooltip.html("positive ratio:" + p_map[key_now])
	            .style("left", d3.event.pageX + "px")
	            .style("top", (d3.event.pageY) + "px")
	            .style("opacity", 0.8);
	    })
	    .on("mousemove", function(d){
	      tooltip.style("left", (d3.event.pageX) + "px")
	            .style("right", (d3.event.pageY) + "px");
	    })
	    .on("mouseout", function(d){
	      tooltip.style("opacity", 0.0);
	    })
	}

	for(var key in t_map){
		//console.log(t_map);
		//console.log("." + key + "_text");
		x_value = d3.select("." + key + "_text")
					.attr("x");
		y_value = d3.select("." + key + "_text")
					.attr("y");
		if(key == 'CA' || key == 'TX' || key == 'FL'){
			x_value -= 90;
		}
		else{
			x_value -= 50;
			y_value -= 70;
		}
		d3.select("body")
    		.append("div")
    		.attr("class", "topic")
    		.style("left", x_value + "px")
    		.style("top", y_value + "px")
    		.html(t_map[key].text);
    	c0 = Number(t_map[key].count_0);
    	c1 = Number(t_map[key].count_1);
    	zero_p = c0 / (c0 + c1);
    	//console.log(c0, c1, zero_p);
    	width = 100 * zero_p;
    	//console.log(zero_p);
    	svg.append("rect")
    		.attr("x", x_value - 30)
    		.attr("y", y_value - 28)
    		.attr("width", width)
    		.attr("height", 20)
    		.attr("fill", "#1e90ff")
    		.attr("opacity", 0.8);
    	svg.append("rect")
    		.attr("x", x_value - 30 + width)
    		.attr("y", y_value - 28)
    		.attr("width", 100 - width)
    		.attr("height", 20)
    		.attr("fill", "#ff0000")
    		.attr("opacity", 0.8);
    	svg.append("text")
    		.html(parseInt(zero_p * 100) + '%')
    		.attr("x", x_value - 25)
    		.attr("y", y_value - 13)
    		.attr("font-size", 13)
    		.attr("fill", "white");
	}

  });

}

function read_data()
{
  d3.csv("static/static_result.csv", function(error, csvdata){
    if(error){
      console.log(error);
    }
    
      var data_map = d3.map(csvdata, function(d){return d.state});
      data_map.forEach(function(key, value){
        p_map[key] = value.p;
      });
      maxvalue = d3.max(csvdata, function(d){
        return d.p;
      });
      minvalue = d3.min(csvdata, function(d){
        return d.p;
      });
    });
}

function read_topic()
{
	d3.csv("static/topic_with_senti.csv", function(error, csvdata){
		if(error){
			console.log(error);
		}
		data_map = d3.map(csvdata, function(d){return d.state});
		data_map.forEach(function(key, value){
			t_map[key] = value;
		});
	});
}

function load_page()
{
  read_data();
  read_topic();
  setTimeout(initial_map(), 200);
  //read_data()
}