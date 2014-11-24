// adapted from https://gist.github.com/magican/5574556

function clone_anchor(thetext, link) {
  // clone link
  var $new_a = $("<a>");
  $new_a.attr("href", link);
  $new_a.text(thetext);
  return $new_a;
}

function ol_depth(element) {
  // get depth of nested ol
  var d = 0;
  while (element.prop("tagName").toLowerCase() == 'ol') {
    d += 1;
    element = element.parent();
  }
  return d;
}

function get_level($e){
  var level=0;
  if ($e.is("h1")){
    level=1;
  } else if ($e.is("h2")){
    level=2;
  } else if ($e.is("h3")){
    level=3;
  } else if ($e.is("h4")){
    level=4;
  } else if ($e.is("h5")){
    level=5;
  } else if ($e.is("h6")){
    level=6;
  }
  else {
    level=0;
  }
  return level
}

function table_of_contents(threshold) {
  if (threshold === undefined) {
    threshold = 5;
  }
  var cells=[];
  $(':header').map(function(i,e){
    var $e = $(e);
    var thetext=$e.text();
    var thelink = $e.find('a.anchor-link').attr("href");
    thetext = thetext.substring(0,thetext.length - 1);
    var level = get_level($e);
    cells.push({"level":level,"thetext":thetext, "thelink":thelink})
    //console.log(thelink, thetext,level);
  });
  console.log("LEVELS", cells);
  var ol = $("<ol/>");
  $("#toc").empty().append(ol);
  
  for (var i=0; i < cells.length; i++) {
    var cell = cells[i];
    
    
    var level = cell.level;
    if (level > threshold) continue;
    console.log("threshold", threshold, level);

    var depth = ol_depth(ol);

    // walk down levels
    for (; depth < level; depth++) {
      var new_ol = $("<ol/>");
      ol.append(new_ol);
      ol = new_ol;
    }
    // walk up levels
    for (; depth > level; depth--) {
      ol = ol.parent();
    }
    //
    ol.append(
      $("<li/>").append(clone_anchor(cell.thetext, cell.thelink))
    );
  }

  $('#toc-wrapper .header').click(function(){
    $('#toc').slideToggle();
    $('#toc-wrapper').toggleClass('closed');
    if ($('#toc-wrapper').hasClass('closed')){
      $('#toc-wrapper .hide-btn').text('[show]');
    } else {
      $('#toc-wrapper .hide-btn').text('[hide]');
    }
    return false;
  })

  $(window).resize(function(){
    $('#toc').css({maxHeight: $(window).height() - 200})
  })

  $(window).trigger('resize');
  $('#toc-wrapper .header').trigger('click');

}

//table_of_contents();


