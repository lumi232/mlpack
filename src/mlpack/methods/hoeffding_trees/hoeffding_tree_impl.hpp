/**
 * @file hoeffding_split_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the HoeffdingTree class.
 */
#ifndef __MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_TREE_IMPL_HPP
#define __MLPACK_METHODS_HOEFFDING_TREES_HOEFFDING_TREE_IMPL_HPP

// In case it hasn't been included yet.
#include "hoeffding_tree.hpp"

namespace mlpack {
namespace tree {

template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType>
template<typename MatType>
HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::HoeffdingTree(const MatType& data,
                 const data::DatasetInfo& datasetInfo,
                 const arma::Row<size_t>& labels,
                 const size_t numClasses,
                 const bool batchTraining,
                 const double successProbability,
                 const size_t maxSamples,
                 const size_t checkInterval) :
    dimensionMappings(new std::unordered_map<size_t,
        std::pair<size_t, size_t>>()),
    ownsMappings(true),
    numSamples(0),
    numClasses(numClasses),
    maxSamples((maxSamples == 0) ? size_t(-1) : maxSamples),
    checkInterval(checkInterval),
    datasetInfo(&datasetInfo),
    ownsInfo(false),
    successProbability(successProbability),
    splitDimension(size_t(-1)),
    categoricalSplit(0),
    numericSplit()
{
  // Generate dimension mappings and create split objects.
  for (size_t i = 0; i < datasetInfo.Dimensionality(); ++i)
  {
    if (datasetInfo.Type(i) == data::Datatype::categorical)
    {
      categoricalSplits.push_back(CategoricalSplitType<FitnessFunction>(
          datasetInfo.NumMappings(i), numClasses));
      (*dimensionMappings)[i] = std::make_pair(data::Datatype::categorical,
          categoricalSplits.size() - 1);
    }
    else
    {
      numericSplits.push_back(NumericSplitType<FitnessFunction>(numClasses));
      (*dimensionMappings)[i] = std::make_pair(data::Datatype::numeric,
          numericSplits.size() - 1);
    }
  }

  // Now train.
  Train(data, labels, batchTraining);
}

template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType>
HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::HoeffdingTree(const data::DatasetInfo& datasetInfo,
                 const size_t numClasses,
                 const double successProbability,
                 const size_t maxSamples,
                 const size_t checkInterval,
                 std::unordered_map<size_t, std::pair<size_t, size_t>>*
                     dimensionMappingsIn) :
    dimensionMappings((dimensionMappingsIn != NULL) ? dimensionMappingsIn :
        new std::unordered_map<size_t, std::pair<size_t, size_t>>()),
    ownsMappings(dimensionMappingsIn == NULL),
    numSamples(0),
    numClasses(numClasses),
    maxSamples((maxSamples == 0) ? size_t(-1) : maxSamples),
    checkInterval(checkInterval),
    datasetInfo(&datasetInfo),
    ownsInfo(false),
    successProbability(successProbability),
    splitDimension(size_t(-1)),
    categoricalSplit(0),
    numericSplit()
{
  // Do we need to generate the mappings too?
  if (ownsMappings)
  {
    for (size_t i = 0; i < datasetInfo.Dimensionality(); ++i)
    {
      if (datasetInfo.Type(i) == data::Datatype::categorical)
      {
        categoricalSplits.push_back(CategoricalSplitType<FitnessFunction>(
            datasetInfo.NumMappings(i), numClasses));
        (*dimensionMappings)[i] = std::make_pair(data::Datatype::categorical,
            categoricalSplits.size() - 1);
      }
      else
      {
        numericSplits.push_back(NumericSplitType<FitnessFunction>(numClasses));
        (*dimensionMappings)[i] = std::make_pair(data::Datatype::numeric,
            numericSplits.size() - 1);
      }
    }
  }
  else
  {
    for (size_t i = 0; i < datasetInfo.Dimensionality(); ++i)
    {
      if (datasetInfo.Type(i) == data::Datatype::categorical)
      {
        categoricalSplits.push_back(CategoricalSplitType<FitnessFunction>(
            datasetInfo.NumMappings(i), numClasses));
      }
      else
      {
        numericSplits.push_back(NumericSplitType<FitnessFunction>(numClasses));
      }
    }
  }
}

// Copy constructor.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType>
HoeffdingTree<FitnessFunction, NumericSplitType, CategoricalSplitType>::
    HoeffdingTree(const HoeffdingTree& other) :
    numericSplits(other.numericSplits),
    categoricalSplits(other.categoricalSplits),
    dimensionMappings(new std::unordered_map<size_t,
        std::pair<size_t, size_t>>(*other.dimensionMappings)),
    ownsMappings(true),
    numSamples(other.numSamples),
    numClasses(other.numClasses),
    maxSamples(other.maxSamples),
    checkInterval(other.checkInterval),
    datasetInfo(new data::DatasetInfo(*other.datasetInfo)),
    ownsInfo(true),
    successProbability(other.successProbability),
    splitDimension(other.splitDimension),
    majorityClass(other.majorityClass),
    majorityProbability(other.majorityProbability),
    categoricalSplit(other.categoricalSplit),
    numericSplit(other.numericSplit),
    children(other.children)
{
  // Nothing left to copy.
}

template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType>
HoeffdingTree<FitnessFunction, NumericSplitType, CategoricalSplitType>::
    ~HoeffdingTree()
{
  if (ownsMappings)
    delete dimensionMappings;
  if (ownsInfo)
    delete datasetInfo;
}

//! Train on a set of points.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType>
template<typename MatType>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::Train(const MatType& data,
         const arma::Row<size_t>& labels,
         const bool batchTraining)
{
  if (batchTraining)
  {
    // Pass all the points through the nodes, and then split only after that.
    checkInterval = data.n_cols; // Only split on the last sample.
    // Don't split if there are fewer than five points.
    size_t oldMaxSamples = maxSamples;
    maxSamples = std::max(size_t(data.n_cols - 1), size_t(5));
    for (size_t i = 0; i < data.n_cols; ++i)
      Train(data.col(i), labels[i]);
    maxSamples = oldMaxSamples;

    // Now, if we did split, find out which points go to which child, and
    // perform the same batch training.
    if (children.size() > 0)
    {
      // We need to create a vector of indices that represent the points that
      // must go to each child, so we need children.size() vectors, but we don't
      // know how long they will be.  Therefore, we will create vectors each of
      // size data.n_cols, but will probably not use all the memory we
      // allocated, and then pass subvectors to the submat() function.
      std::vector<arma::uvec> indices(children.size(), arma::uvec(data.n_cols));
      arma::Col<size_t> counts =
          arma::zeros<arma::Col<size_t>>(children.size());

      for (size_t i = 0; i < data.n_cols; ++i)
      {
        size_t direction = CalculateDirection(data.col(i));
        size_t currentIndex = counts[direction];
        indices[direction][currentIndex] = i;
        counts[direction]++;
      }

      // Now pass each of these submatrices to the children to perform
      // batch-mode training.
      for (size_t i = 0; i < children.size(); ++i)
      {
        // If we don't have any points that go to the child in question, don't
        // train that child.
        if (counts[i] == 0)
          continue;

        // The submatrix here is non-contiguous, but I think this will be faster
        // than copying the points to an ordered state.  We still have to
        // assemble the labels vector, though.
        arma::Row<size_t> childLabels = labels.cols(
            indices[i].subvec(0, counts[i] - 1));

        // Unfortunately, limitations of Armadillo's non-contiguous subviews
        // prohibits us from successfully passing the non-contiguous subview to
        // Train(), since the col() function is not provided.  So,
        // unfortunately, instead, we'll just extract the non-contiguous
        // submatrix.
        MatType childData = data.cols(indices[i].subvec(0, counts[i] - 1));
        children[i].Train(childData, childLabels, true);
      }
    }
  }
  else
  {
    // We aren't training in batch mode; loop through the points.
    for (size_t i = 0; i < data.n_cols; ++i)
      Train(data.col(i), labels[i]);
  }
}

//! Train on one point.
template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType>
template<typename VecType>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::Train(const VecType& point, const size_t label)
{
  if (splitDimension == size_t(-1))
  {
    ++numSamples;
    size_t numericIndex = 0;
    size_t categoricalIndex = 0;
    for (size_t i = 0; i < point.n_rows; ++i)
    {
      if (datasetInfo->Type(i) == data::Datatype::categorical)
        categoricalSplits[categoricalIndex++].Train(point[i], label);
      else if (datasetInfo->Type(i) == data::Datatype::numeric)
        numericSplits[numericIndex++].Train(point[i], label);
    }

    // Grab majority class from splits.
    if (categoricalSplits.size() > 0)
    {
      majorityClass = categoricalSplits[0].MajorityClass();
      majorityProbability = categoricalSplits[0].MajorityProbability();
    }
    else
    {
      majorityClass = numericSplits[0].MajorityClass();
      majorityProbability = numericSplits[0].MajorityProbability();
    }

    // Check for a split, if we should.
    if (numSamples % checkInterval == 0)
    {
      const size_t numChildren = SplitCheck();
      if (numChildren > 0)
      {
        // We need to add a bunch of children.
        // Delete children, if we have them.
        children.clear();
        CreateChildren();
      }
    }
  }
  else
  {
    // Already split.  Pass the training point to the relevant child.
    size_t direction = CalculateDirection(point);
    children[direction].Train(point, label);
  }
}

template<typename FitnessFunction,
         template<typename> class NumericSplitType,
         template<typename> class CategoricalSplitType>
size_t HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::SplitCheck()
{
  // Do nothing if we've already split.
  if (splitDimension != size_t(-1))
    return 0;

  // Check the fitness of each dimension.  Then we'll use a Hoeffding bound
  // somehow.

  // Calculate epsilon, the value we need things to be greater than.
  const double rSquared = std::pow(FitnessFunction::Range(numClasses), 2.0);
  const double epsilon = std::sqrt(rSquared *
      std::log(1.0 / (1.0 - successProbability)) / (2 * numSamples));

  arma::vec gains(categoricalSplits.size() + numericSplits.size());
  for (size_t i = 0; i < gains.n_elem; ++i)
  {
    size_t type = dimensionMappings->at(i).first;
    size_t index = dimensionMappings->at(i).second;
    if (type == data::Datatype::categorical)
      gains[i] = categoricalSplits[index].EvaluateFitnessFunction();
    else if (type == data::Datatype::numeric)
      gains[i] = numericSplits[index].EvaluateFitnessFunction();
  }

  // Now find the largest and second-largest.
  double largest = -DBL_MAX;
  size_t largestIndex = 0;
  double secondLargest = -DBL_MAX;
  for (size_t i = 0; i < gains.n_elem; ++i)
  {
    if (gains[i] > largest)
    {
      secondLargest = largest;
      largest = gains[i];
      largestIndex = i;
    }
    else if (gains[i] > secondLargest)
    {
      secondLargest = gains[i];
    }
  }

  // Are these far enough apart to split?
  if ((largest - secondLargest > epsilon) ||
      ((numSamples > maxSamples) && (largest > 0.0)))
  {
    // Split!
    splitDimension = largestIndex;
    const size_t type = dimensionMappings->at(largestIndex).first;
    const size_t index = dimensionMappings->at(largestIndex).second;
    if (type == data::Datatype::categorical)
    {
      // I don't know if this should be here.
      majorityClass = categoricalSplits[index].MajorityClass();
      return categoricalSplits[index].NumChildren();
    }
    else
    {
      majorityClass = numericSplits[index].MajorityClass();
      return numericSplits[index].NumChildren();
    }
  }
  else
  {
    return 0; // Don't split.
  }
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename VecType>
size_t HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::CalculateDirection(const VecType& point) const
{
  // Don't call this before the node is split...
  if (datasetInfo->Type(splitDimension) == data::Datatype::numeric)
    return numericSplit.CalculateDirection(point[splitDimension]);
  else if (datasetInfo->Type(splitDimension) == data::Datatype::categorical)
    return categoricalSplit.CalculateDirection(point[splitDimension]);
  else
    return 0; // Not sure what to do here...
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename VecType>
size_t HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::Classify(const VecType& point) const
{
  if (children.size() == 0)
  {
    // If we're a leaf (or being considered a leaf), classify based on what we
    // know.
    return majorityClass;
  }
  else
  {
    // Otherwise, pass to the right child and let them classify.
    return children[CalculateDirection(point)].Classify(point);
  }
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename VecType>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::Classify(const VecType& point,
            size_t& prediction,
            double& probability) const
{
  if (children.size() == 0)
  {
    // We are a leaf, so classify accordingly.
    prediction = majorityClass;
    probability = majorityProbability;
  }
  else
  {
    // Pass to the right child and let them do the classification.
    children[CalculateDirection(point)].Classify(point, prediction,
        probability);
  }
}

//! Batch classification.
template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename MatType>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::Classify(const MatType& data, arma::Row<size_t>& predictions) const
{
  predictions.set_size(data.n_cols);
  for (size_t i = 0; i < data.n_cols; ++i)
    predictions[i] = Classify(data.col(i));
}

//! Batch classification with probabilities.
template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename MatType>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::Classify(const MatType& data,
            arma::Row<size_t>& predictions,
            arma::rowvec& probabilities) const
{
  predictions.set_size(data.n_cols);
  probabilities.set_size(data.n_cols);
  for (size_t i = 0; i < data.n_cols; ++i)
    Classify(data.col(i), predictions[i], probabilities[i]);
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::CreateChildren()
{
  // Create the children.
  arma::Col<size_t> childMajorities;
  if (dimensionMappings->at(splitDimension).first ==
      data::Datatype::categorical)
  {
    categoricalSplits[dimensionMappings->at(splitDimension).second].Split(
        childMajorities, categoricalSplit);
  }
  else if (dimensionMappings->at(splitDimension).first ==
           data::Datatype::numeric)
  {
    numericSplits[dimensionMappings->at(splitDimension).second].Split(
        childMajorities, numericSplit);
  }

  // We already know what the splitDimension will be.
  for (size_t i = 0; i < childMajorities.n_elem; ++i)
  {
    children.push_back(HoeffdingTree(*datasetInfo, numClasses,
        successProbability, maxSamples, checkInterval, dimensionMappings));
    children[i].MajorityClass() = childMajorities[i];
  }

  // Eliminate now-unnecessary split information.
  numericSplits.clear();
  categoricalSplits.clear();
}

template<
    typename FitnessFunction,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename Archive>
void HoeffdingTree<
    FitnessFunction,
    NumericSplitType,
    CategoricalSplitType
>::Serialize(Archive& ar, const unsigned int /* version */)
{
  using data::CreateNVP;

  ar & CreateNVP(splitDimension, "splitDimension");
  ar & CreateNVP(dimensionMappings, "dimensionMappings");
  ar & CreateNVP(ownsMappings, "ownsMappings");

  // Special handling for const object.
  data::DatasetInfo* d = NULL;
  if (Archive::is_saving::value)
    d = const_cast<data::DatasetInfo*>(datasetInfo);
  ar & CreateNVP(d, "datasetInfo");
  if (Archive::is_loading::value)
  {
    datasetInfo = d;
    ownsInfo = true;
  }

  ar & CreateNVP(majorityClass, "majorityClass");
  ar & CreateNVP(majorityProbability, "majorityProbability");

  // Depending on whether or not we have split yet, we may need to save
  // different things.
  if (splitDimension == size_t(-1))
  {
    // We have not yet split.  So we have to serialize the splits.
    ar & CreateNVP(numSamples, "numSamples");
    ar & CreateNVP(numClasses, "numClasses");
    ar & CreateNVP(maxSamples, "maxSamples");
    ar & CreateNVP(successProbability, "successProbability");

    // Serialize the splits, but not if we haven't seen any samples yet (in
    // which case we can just reinitialize).
    if (Archive::is_loading::value)
    {
      // Re-initialize all of the splits.
      numericSplits.clear();
      categoricalSplits.clear();
      for (size_t i = 0; i < datasetInfo->Dimensionality(); ++i)
      {
        if (datasetInfo->Type(i) == data::Datatype::categorical)
          categoricalSplits.push_back(CategoricalSplitType<FitnessFunction>(
              datasetInfo->NumMappings(i), numClasses));
        else
          numericSplits.push_back(
              NumericSplitType<FitnessFunction>(numClasses));
      }

      // Clear things we don't need.
      categoricalSplit = typename CategoricalSplitType<FitnessFunction>::
          SplitInfo(numClasses);
      numericSplit = typename NumericSplitType<FitnessFunction>::SplitInfo();
    }

    // There's no need to serialize if there's no information contained in the
    // splits.
    if (numSamples == 0)
      return;

    // Serialize numeric splits.
    for (size_t i = 0; i < numericSplits.size(); ++i)
    {
      std::ostringstream name;
      name << "numericSplit" << i;
      ar & CreateNVP(numericSplits[i], name.str());
    }

    // Serialize categorical splits.
    for (size_t i = 0; i < categoricalSplits.size(); ++i)
    {
      std::ostringstream name;
      name << "categoricalSplit" << i;
      ar & CreateNVP(categoricalSplits[i], name.str());
    }
  }
  else
  {
    // We have split, so we only need to save the split and the children.
    if (datasetInfo->Type(splitDimension) == data::Datatype::categorical)
      ar & CreateNVP(categoricalSplit, "categoricalSplit");
    else
      ar & CreateNVP(numericSplit, "numericSplit");

    // Serialize the children, because we have split.
    size_t numChildren;
    if (Archive::is_saving::value)
      numChildren = children.size();
    ar & CreateNVP(numChildren, "numChildren");
    if (Archive::is_loading::value) // If needed, allocate space.
      children.resize(numChildren, HoeffdingTree(data::DatasetInfo(0), 0));

    for (size_t i = 0; i < numChildren; ++i)
    {
      std::ostringstream name;
      name << "child" << i;
      ar & data::CreateNVP(children[i], name.str());
    }

    if (Archive::is_loading::value)
    {
      numericSplits.clear();
      categoricalSplits.clear();

      numSamples = 0;
      numClasses = 0;
      maxSamples = 0;
      successProbability = 0.0;
    }
  }
}

} // namespace tree
} // namespace mlpack

#endif