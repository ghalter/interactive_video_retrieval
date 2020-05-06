var currentResult = []
var previousHistory = []

function pushHistory(type="Query"){
    t = []
    currentResult.forEach((elem) =>{
        t.push(elem)
    })
    previousHistory.push([type, t])
    updateHistoryView();
}

function updateHistoryView(){
    $("#history-list").html("")
    let idx = 0;
    previousHistory.forEach((elem) => {
        // var idx = previousHistory.length - 1;
        var name = 'btn-history-'+ idx;
        var html =  '<li class="list-group-item list-group-flush"><button class="btn btn-link collapsed" id="'+name + '">'+elem[0]+': </button></li>'
        $("#history-list").prepend(html)
        let nd = idx
        $("#"+name).on("click", function(){
            loadHistoryPoint(nd)
        })
        idx ++;
    })
}

function loadHistoryPoint(idx){
    currentResult = previousHistory[idx][1]
    updateResultView();
}
