// append the svg object to the body of the page


var data = []
// for (var i = 0; i<1000; i++){
//     data.push({
//         x: Math.random() * 1000,
//         y: Math.random() * 1000,
//         url: "https://live.staticflickr.com/7836/47449269212_81b5f62fdd_q.jpg"
//     })
// }

let cwidth = $("#svgwrapper-grid").css("width").replace("px", "")
let r = (cwidth / 20)
var j = 0
function imageGrid() {
    for (var i = 0; i < 20; i++) {
        for (j = 0; j < 100; j++) {
            data.push({
                x: i * r,
                y: j * r,
                url: "https://live.staticflickr.com/7836/47449269212_81b5f62fdd_q.jpg"
            })
        }
    }


}

function updateImageGrid(result) {
    let counter = 0;
    data.length = 0;
    for (j = 0; j < 100; j++) {
    for (var i = 0; i < 20; i++) {
            if (counter < result.length){
                data.push({
                    x: i * r,
                    y: j * r,
                    url: result[counter].thumbnail
                })
            }
            counter++;
        }
    }
    updatePlot();
}

var zoom = d3.zoom()
    .scaleExtent([1.0, 5])
    // .translateExtent([[ $("#svgwrapper-grid").position().left, $("#svgwrapper-grid").position().top], [20 * r, j * r]])
    // // .extent([[0,0],[20 * r, j * r]])
    .on("zoom", function () {
        svg.attr("transform", d3.event.transform)
    })

imageGrid()
var svg = d3.select("#svgwrapper-grid")
    .append("svg")
    .attr("width", "100%")
    .attr("height", "100%")
    .call(zoom);

updatePlot();

function updatePlot() {
    svg.selectAll("image").remove();
    svg.selectAll("rect").remove();

    svg.append("g")
        .selectAll("image")
        .data(data)
        .enter()
        .append("image")
        .attr("x", function (d) { return d.x })
        .attr("y", function (d) { return d.y })
        .attr("width", r)
        .attr("height", r)
        .attr("xlink:href", function (d) { return d.url; })

    svg.append("g")
        .selectAll("rect")
        .data(data)
        .enter()
        .append("rect")
        .attr("x", function (d) { return d.x })
        .attr("y", function (d) { return d.y })
        .attr("width", r)
        .attr("height", r)
        .attr('fill', 'rgba(0,0,0,0.4)')
        .attr('stroke', '#FFF')
        .attr('stroke-width', '3')
        .on("mouseover", handleMouseOver)
        .on("mouseout", handleMouseOut);



    function handleMouseOver(d, i) {
        d3.select(this).raise();
        d3.select(this)
            .attr('fill', 'rgba(0,0,0,0)')
            .attr('stroke', 'orange')
            .attr('stroke-width', '3');

    }
    function handleMouseOut(d, i) {
        d3.select(this)
            .attr('fill', 'rgba(0,0,0, 0.4)')
            .attr('stroke', '#FFF')
            .attr('stroke-width', '3');
    }

}
